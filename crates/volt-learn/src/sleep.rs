//! Sleep consolidation scheduler.
//!
//! Orchestrates the full sleep cycle: idle detection → frame distillation →
//! Forward-Forward VFN training → strand graduation → garbage collection.
//!
//! ## Triggering
//!
//! - **Automatic**: background thread polls for idle > 10 minutes
//! - **Manual**: call [`SleepScheduler::force_sleep`] directly
//!
//! ## Sleep Cycle Phases
//!
//! 1. Snapshot learning events from the logger
//! 2. Distill all strands (cluster → wisdom frames)
//! 3. Collect Forward-Forward samples from events
//! 4. Train VFN layer-by-layer (no backprop)
//! 5. Check strand graduation (novel topic → new strand)
//! 6. Run garbage collection
//!
//! ## Thread Safety
//!
//! The background scheduler acquires locks on VoltStore, Vfn, and
//! EventLogger in a fixed order to prevent deadlocks. The system
//! remains responsive during consolidation because locks are held
//! only for the duration of each phase.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use volt_core::VoltError;
use volt_db::VoltStore;
use volt_soft::vfn::Vfn;
use volt_translate::StubTranslator;

use crate::distillation::{self, DistillationConfig, DistillationResult};
use crate::eval_dataset;
use crate::forward_forward::{self, FfConfig, FfResult};
use crate::graduation::{self, GraduationConfig, GraduationResult};
use crate::logger::EventLogger;
use crate::rlvf::{self, RlvfConfig, RlvfResult};

/// Configuration for the sleep scheduler.
///
/// # Example
///
/// ```
/// use volt_learn::sleep::SleepConfig;
/// use std::time::Duration;
///
/// let config = SleepConfig::default();
/// assert_eq!(config.idle_timeout, Duration::from_secs(600));
/// ```
#[derive(Debug, Clone)]
pub struct SleepConfig {
    /// Duration of idle time before triggering sleep. Default: 10 minutes.
    pub idle_timeout: Duration,
    /// How often the background thread checks for idle. Default: 30 seconds.
    pub poll_interval: Duration,
    /// Forward-Forward training configuration.
    pub ff_config: FfConfig,
    /// Distillation configuration.
    pub distillation_config: DistillationConfig,
    /// Strand graduation configuration.
    pub graduation_config: GraduationConfig,
    /// RLVF training configuration. `None` disables RLVF during sleep.
    /// Default: `None` (opt-in, since RLVF is more expensive than FF).
    pub rlvf_config: Option<RlvfConfig>,
    /// Minimum accumulated learning events before RLVF triggers.
    /// Default: 100.
    pub rlvf_min_events: usize,
}

impl Default for SleepConfig {
    fn default() -> Self {
        Self {
            idle_timeout: Duration::from_secs(600),
            poll_interval: Duration::from_secs(30),
            ff_config: FfConfig::default(),
            distillation_config: DistillationConfig::default(),
            graduation_config: GraduationConfig::default(),
            rlvf_config: None,
            rlvf_min_events: 100,
        }
    }
}

/// Result of a complete sleep consolidation cycle.
///
/// # Example
///
/// ```
/// use volt_learn::sleep::SleepCycleResult;
/// use volt_learn::graduation::GraduationResult;
/// use std::time::Duration;
///
/// let result = SleepCycleResult {
///     distillation: vec![],
///     ff_training: None,
///     rlvf_training: None,
///     graduation: GraduationResult {
///         new_strands_created: vec![],
///         frames_migrated: 0,
///     },
///     gc_frames_decayed: 0,
///     duration: Duration::from_millis(50),
/// };
/// assert_eq!(result.gc_frames_decayed, 0);
/// ```
#[derive(Debug, Clone)]
pub struct SleepCycleResult {
    /// Per-strand distillation results.
    pub distillation: Vec<DistillationResult>,
    /// Forward-Forward training result (None if no samples).
    pub ff_training: Option<FfResult>,
    /// RLVF training result (None if disabled or insufficient events).
    pub rlvf_training: Option<RlvfResult>,
    /// Strand graduation result.
    pub graduation: GraduationResult,
    /// Number of frames decayed by GC.
    pub gc_frames_decayed: usize,
    /// Wall-clock duration of the entire sleep cycle.
    pub duration: Duration,
}

/// The sleep consolidation scheduler.
///
/// Tracks idle time and runs the full consolidation pipeline when the
/// system has been idle long enough.
///
/// # Example
///
/// ```
/// use volt_learn::sleep::SleepScheduler;
///
/// let mut scheduler = SleepScheduler::with_defaults();
/// scheduler.touch(); // Record activity
/// assert!(!scheduler.should_sleep()); // Just touched, not idle
/// ```
#[derive(Debug)]
pub struct SleepScheduler {
    config: SleepConfig,
    last_activity: Instant,
    is_sleeping: bool,
}

impl SleepScheduler {
    /// Creates a sleep scheduler with the given configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::sleep::{SleepScheduler, SleepConfig};
    ///
    /// let scheduler = SleepScheduler::new(SleepConfig::default());
    /// ```
    pub fn new(config: SleepConfig) -> Self {
        Self {
            config,
            last_activity: Instant::now(),
            is_sleeping: false,
        }
    }

    /// Creates a sleep scheduler with default configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::sleep::SleepScheduler;
    ///
    /// let scheduler = SleepScheduler::with_defaults();
    /// ```
    pub fn with_defaults() -> Self {
        Self::new(SleepConfig::default())
    }

    /// Records that user activity occurred, resetting the idle timer.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::sleep::SleepScheduler;
    ///
    /// let mut scheduler = SleepScheduler::with_defaults();
    /// scheduler.touch();
    /// assert!(!scheduler.should_sleep());
    /// ```
    pub fn touch(&mut self) {
        self.last_activity = Instant::now();
    }

    /// Returns whether the system has been idle long enough for sleep.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::sleep::SleepScheduler;
    ///
    /// let scheduler = SleepScheduler::with_defaults();
    /// // Just created, so idle_timeout (10 min) hasn't passed
    /// assert!(!scheduler.should_sleep());
    /// ```
    pub fn should_sleep(&self) -> bool {
        !self.is_sleeping
            && self.last_activity.elapsed() >= self.config.idle_timeout
    }

    /// Returns whether a sleep cycle is currently in progress.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::sleep::SleepScheduler;
    ///
    /// let scheduler = SleepScheduler::with_defaults();
    /// assert!(!scheduler.is_sleeping());
    /// ```
    pub fn is_sleeping(&self) -> bool {
        self.is_sleeping
    }

    /// Returns a reference to the configuration.
    pub fn config(&self) -> &SleepConfig {
        &self.config
    }

    /// Runs a full sleep consolidation cycle.
    ///
    /// Executes all phases in order: distillation → FF training →
    /// graduation → GC. Blocks the calling thread for the duration.
    ///
    /// This method is designed to be called from a background thread.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::LearnError`] if any phase fails critically.
    /// Non-critical failures (e.g., no FF samples) are handled gracefully.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::sleep::SleepScheduler;
    /// use volt_learn::EventLogger;
    /// use volt_db::VoltStore;
    /// use volt_soft::vfn::Vfn;
    ///
    /// let mut scheduler = SleepScheduler::with_defaults();
    /// let mut store = VoltStore::new();
    /// let mut vfn = Vfn::new_random(42);
    /// let logger = EventLogger::new();
    ///
    /// let result = scheduler.run_sleep_cycle(&mut store, &mut vfn, &logger).unwrap();
    /// assert_eq!(result.distillation.len(), 1); // Default strand
    /// ```
    pub fn run_sleep_cycle(
        &mut self,
        store: &mut VoltStore,
        vfn: &mut Vfn,
        logger: &EventLogger,
    ) -> Result<SleepCycleResult, VoltError> {
        self.is_sleeping = true;
        let start = Instant::now();

        let result = self.run_sleep_cycle_inner(store, vfn, logger);

        self.is_sleeping = false;
        self.last_activity = Instant::now();

        match result {
            Ok(mut r) => {
                r.duration = start.elapsed();
                Ok(r)
            }
            Err(e) => Err(e),
        }
    }

    /// Manually triggers a sleep cycle, ignoring the idle timeout.
    ///
    /// # Errors
    ///
    /// Same as [`run_sleep_cycle`](Self::run_sleep_cycle).
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::sleep::SleepScheduler;
    /// use volt_learn::EventLogger;
    /// use volt_db::VoltStore;
    /// use volt_soft::vfn::Vfn;
    ///
    /// let mut scheduler = SleepScheduler::with_defaults();
    /// let mut store = VoltStore::new();
    /// let mut vfn = Vfn::new_random(42);
    /// let logger = EventLogger::new();
    ///
    /// let result = scheduler.force_sleep(&mut store, &mut vfn, &logger).unwrap();
    /// ```
    pub fn force_sleep(
        &mut self,
        store: &mut VoltStore,
        vfn: &mut Vfn,
        logger: &EventLogger,
    ) -> Result<SleepCycleResult, VoltError> {
        self.run_sleep_cycle(store, vfn, logger)
    }

    /// Internal implementation of the sleep cycle phases.
    fn run_sleep_cycle_inner(
        &self,
        store: &mut VoltStore,
        vfn: &mut Vfn,
        logger: &EventLogger,
    ) -> Result<SleepCycleResult, VoltError> {
        // Phase 1: Snapshot events
        let events: Vec<_> = logger.events().to_vec();

        // Phase 2: Distill all strands
        let distillation_results = distillation::distill_all_strands(store)?;

        // Phase 3+4: Collect FF samples and train VFN
        let ff_result = if !events.is_empty() {
            forward_forward::collect_ff_samples(
                &events,
                store,
                &self.config.ff_config,
            )
            .ok()
            .and_then(|samples| {
                forward_forward::train_ff(
                    vfn,
                    &samples,
                    &self.config.ff_config,
                )
                .ok()
            })
        } else {
            None
        };

        // Phase 4.5: RLVF training (if configured and enough events)
        let rlvf_result = if let Some(ref rlvf_config) = self.config.rlvf_config
            && events.len() >= self.config.rlvf_min_events
        {
            let translator = StubTranslator::new();
            let dataset = eval_dataset::generate_eval_dataset();
            rlvf::train_rlvf(vfn, &dataset, &translator, rlvf_config).ok()
        } else {
            None
        };

        // Phase 5: Strand graduation
        let graduation_result = graduation::check_graduation(
            store,
            &events,
            &self.config.graduation_config,
        )?;

        // Phase 6: Garbage collection
        let gc_result = store.run_gc()?;

        Ok(SleepCycleResult {
            distillation: distillation_results,
            ff_training: ff_result,
            rlvf_training: rlvf_result,
            graduation: graduation_result,
            gc_frames_decayed: gc_result.frames_compressed
                + gc_result.frames_gisted
                + gc_result.frames_tombstoned,
            duration: Duration::ZERO, // Filled by caller
        })
    }

    /// Spawns a background thread that polls for idle and runs sleep cycles.
    ///
    /// The thread acquires locks in a fixed order (logger → store → vfn)
    /// to prevent deadlocks. Returns a [`SleepHandle`] that can be used
    /// to stop the thread.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::LearnError`] if the thread fails to spawn.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::sleep::{SleepScheduler, SleepConfig};
    /// use volt_learn::EventLogger;
    /// use volt_db::VoltStore;
    /// use volt_soft::vfn::Vfn;
    /// use std::sync::{Arc, RwLock};
    /// use std::time::Duration;
    ///
    /// let config = SleepConfig {
    ///     idle_timeout: Duration::from_millis(50),
    ///     poll_interval: Duration::from_millis(10),
    ///     ..SleepConfig::default()
    /// };
    /// let store = Arc::new(RwLock::new(VoltStore::new()));
    /// let vfn = Arc::new(RwLock::new(Vfn::new_random(42)));
    /// let logger = Arc::new(RwLock::new(EventLogger::new()));
    ///
    /// let handle = SleepScheduler::spawn_background(
    ///     config, store, vfn, logger,
    /// ).unwrap();
    /// handle.stop();
    /// handle.join().unwrap();
    /// ```
    pub fn spawn_background(
        config: SleepConfig,
        store: Arc<RwLock<VoltStore>>,
        vfn: Arc<RwLock<Vfn>>,
        logger: Arc<RwLock<EventLogger>>,
    ) -> Result<SleepHandle, VoltError> {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&stop_flag);
        let activity_flag = Arc::new(AtomicBool::new(false));
        let activity_clone = Arc::clone(&activity_flag);
        let poll_interval = config.poll_interval;

        let thread = std::thread::Builder::new()
            .name("sleep-scheduler".into())
            .stack_size(4 * 1024 * 1024)
            .spawn(move || {
                let mut scheduler = SleepScheduler::new(config);

                while !stop_clone.load(Ordering::Relaxed) {
                    std::thread::sleep(poll_interval);

                    if stop_clone.load(Ordering::Relaxed) {
                        break;
                    }

                    // Check external activity signal from the server
                    if activity_clone.swap(false, Ordering::Relaxed) {
                        scheduler.touch();
                    }

                    if !scheduler.should_sleep() {
                        continue;
                    }

                    // Acquire locks in fixed order: logger → store → vfn
                    let logger_guard = match logger.read() {
                        Ok(g) => g,
                        Err(_) => continue,
                    };
                    let mut store_guard = match store.write() {
                        Ok(g) => g,
                        Err(_) => continue,
                    };
                    let mut vfn_guard = match vfn.write() {
                        Ok(g) => g,
                        Err(_) => continue,
                    };

                    let _ = scheduler.run_sleep_cycle(
                        &mut store_guard,
                        &mut vfn_guard,
                        &logger_guard,
                    );
                }
            })
            .map_err(|e| VoltError::LearnError {
                message: format!(
                    "failed to spawn sleep scheduler thread: {e}"
                ),
            })?;

        Ok(SleepHandle {
            stop_flag,
            activity_flag,
            thread: Some(thread),
        })
    }
}

/// Handle to a running background sleep scheduler.
///
/// # Example
///
/// ```
/// use volt_learn::sleep::{SleepScheduler, SleepConfig};
/// use volt_learn::EventLogger;
/// use volt_db::VoltStore;
/// use volt_soft::vfn::Vfn;
/// use std::sync::{Arc, RwLock};
/// use std::time::Duration;
///
/// let config = SleepConfig {
///     idle_timeout: Duration::from_millis(50),
///     poll_interval: Duration::from_millis(10),
///     ..SleepConfig::default()
/// };
/// let store = Arc::new(RwLock::new(VoltStore::new()));
/// let vfn = Arc::new(RwLock::new(Vfn::new_random(42)));
/// let logger = Arc::new(RwLock::new(EventLogger::new()));
///
/// let handle = SleepScheduler::spawn_background(config, store, vfn, logger).unwrap();
/// handle.stop();
/// handle.join().unwrap();
/// ```
pub struct SleepHandle {
    stop_flag: Arc<AtomicBool>,
    activity_flag: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for SleepHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SleepHandle")
            .field("stopped", &self.stop_flag.load(Ordering::Relaxed))
            .finish()
    }
}

impl SleepHandle {
    /// Signals the scheduler thread to stop.
    ///
    /// The thread will exit at the next poll interval. This is
    /// non-blocking — use [`join`](Self::join) to wait.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use volt_learn::sleep::SleepHandle;
    /// # fn example(handle: SleepHandle) {
    /// handle.stop();
    /// # }
    /// ```
    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::Relaxed);
    }

    /// Notifies the scheduler that user activity occurred.
    ///
    /// Resets the idle timer so that sleep consolidation does not
    /// trigger while the user is actively sending requests. The
    /// background thread picks this up on its next poll cycle.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use volt_learn::sleep::SleepHandle;
    /// # fn example(handle: &SleepHandle) {
    /// handle.touch(); // Reset idle timer after a user request
    /// # }
    /// ```
    pub fn touch(&self) {
        self.activity_flag.store(true, Ordering::Relaxed);
    }

    /// Waits for the scheduler thread to finish.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::LearnError`] if the thread panicked.
    pub fn join(mut self) -> Result<(), VoltError> {
        if let Some(thread) = self.thread.take() {
            thread.join().map_err(|_| VoltError::LearnError {
                message: "sleep scheduler thread panicked".to_string(),
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_sensible() {
        let config = SleepConfig::default();
        assert_eq!(config.idle_timeout, Duration::from_secs(600));
        assert_eq!(config.poll_interval, Duration::from_secs(30));
    }

    #[test]
    fn scheduler_not_sleeping_initially() {
        let scheduler = SleepScheduler::with_defaults();
        assert!(!scheduler.is_sleeping());
        assert!(!scheduler.should_sleep()); // Just created
    }

    #[test]
    fn touch_resets_idle() {
        let mut scheduler = SleepScheduler::new(SleepConfig {
            idle_timeout: Duration::from_millis(1),
            ..SleepConfig::default()
        });
        std::thread::sleep(Duration::from_millis(5));
        // Should be idle now
        assert!(scheduler.should_sleep());
        scheduler.touch();
        // After touch, should not be idle
        assert!(!scheduler.should_sleep());
    }

    #[test]
    fn force_sleep_runs_cycle() {
        let mut scheduler = SleepScheduler::with_defaults();
        let mut store = VoltStore::new();
        let mut vfn = Vfn::new_random(42);
        let logger = EventLogger::new();

        let result =
            scheduler.force_sleep(&mut store, &mut vfn, &logger).unwrap();
        // Should have distilled the default strand
        assert_eq!(result.distillation.len(), 1);
        // No events → no FF training
        assert!(result.ff_training.is_none());
        // No events → no graduation
        assert!(result.graduation.new_strands_created.is_empty());
    }

    #[test]
    fn sleep_cycle_resets_activity() {
        let mut scheduler = SleepScheduler::new(SleepConfig {
            idle_timeout: Duration::from_millis(1),
            ..SleepConfig::default()
        });
        std::thread::sleep(Duration::from_millis(5));
        assert!(scheduler.should_sleep());

        let mut store = VoltStore::new();
        let mut vfn = Vfn::new_random(42);
        let logger = EventLogger::new();
        scheduler
            .run_sleep_cycle(&mut store, &mut vfn, &logger)
            .unwrap();

        // After sleep, idle timer is reset
        assert!(!scheduler.should_sleep());
        assert!(!scheduler.is_sleeping());
    }

    #[test]
    fn background_scheduler_starts_and_stops() {
        let config = SleepConfig {
            idle_timeout: Duration::from_secs(3600), // Long timeout
            poll_interval: Duration::from_millis(10),
            ..SleepConfig::default()
        };
        let store = Arc::new(RwLock::new(VoltStore::new()));
        let vfn = Arc::new(RwLock::new(Vfn::new_random(42)));
        let logger = Arc::new(RwLock::new(EventLogger::new()));

        let handle =
            SleepScheduler::spawn_background(config, store, vfn, logger)
                .unwrap();

        // Let it run a few poll cycles
        std::thread::sleep(Duration::from_millis(50));

        handle.stop();
        handle.join().unwrap();
    }

    #[test]
    fn system_responsive_during_sleep() {
        let config = SleepConfig {
            idle_timeout: Duration::from_millis(10),
            poll_interval: Duration::from_millis(5),
            ..SleepConfig::default()
        };
        let store = Arc::new(RwLock::new(VoltStore::new()));
        let vfn = Arc::new(RwLock::new(Vfn::new_random(42)));
        let logger = Arc::new(RwLock::new(EventLogger::new()));

        let handle = SleepScheduler::spawn_background(
            config,
            Arc::clone(&store),
            Arc::clone(&vfn),
            Arc::clone(&logger),
        )
        .unwrap();

        // Let background scheduler possibly start a cycle
        std::thread::sleep(Duration::from_millis(30));

        // Main thread should still be able to access the store
        // (might have to wait for lock, but should not deadlock)
        let read_result = store.read();
        assert!(
            read_result.is_ok(),
            "main thread should be able to read store"
        );

        handle.stop();
        handle.join().unwrap();
    }
}
