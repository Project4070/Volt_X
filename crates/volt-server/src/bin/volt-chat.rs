//! Interactive CLI chat client for Volt X.
//!
//! Connects to the Volt X server HTTP API and provides a REPL
//! with conversation tracking, debug mode, and introspection.
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin volt-chat
//! cargo run --bin volt-chat -- --url http://localhost:8080
//! cargo run --bin volt-chat -- --conversation 123
//! ```

use std::error::Error;

use reqwest::blocking::Client;
use rustyline::error::ReadlineError;
use rustyline::Editor;
use volt_server::models::{
    ConversationListResponse, ConversationMeta, CreateConversationResponse, ThinkRequest,
    ThinkResponse,
};

/// Main chat client state
struct ChatClient {
    /// Base URL of the Volt X server
    base_url: String,
    /// HTTP client for API requests
    client: Client,
    /// Current conversation ID (None = will auto-create on first message)
    conversation_id: Option<u64>,
    /// Whether to show debug information
    debug_mode: bool,
}

impl ChatClient {
    /// Create a new chat client
    fn new(url: String) -> Self {
        Self {
            base_url: url,
            client: Client::new(),
            conversation_id: None,
            debug_mode: false,
        }
    }

    /// Send a message to the server and get a response
    fn send_message(&mut self, text: &str) -> Result<ThinkResponse, Box<dyn Error>> {
        let url = format!("{}/api/think", self.base_url);
        let request = ThinkRequest {
            text: text.to_string(),
            conversation_id: self.conversation_id,
        };

        let response = self.client.post(&url).json(&request).send()?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().unwrap_or_else(|_| "unknown error".to_string());
            return Err(format!("HTTP {status}: {error_text}").into());
        }

        let think_response: ThinkResponse = response.json()?;

        // Update conversation_id if it was auto-created
        if self.conversation_id.is_none() {
            self.conversation_id = Some(think_response.conversation_id);
        }

        Ok(think_response)
    }

    /// Create a new conversation
    #[allow(dead_code)]
    fn create_conversation(&mut self) -> Result<u64, Box<dyn Error>> {
        let url = format!("{}/api/conversations", self.base_url);
        let response = self.client.post(&url).send()?;

        if !response.status().is_success() {
            let status = response.status();
            return Err(format!("Failed to create conversation: HTTP {status}").into());
        }

        let create_response: CreateConversationResponse = response.json()?;
        Ok(create_response.conversation_id)
    }

    /// List all conversations
    fn list_conversations(&self) -> Result<Vec<ConversationMeta>, Box<dyn Error>> {
        let url = format!("{}/api/conversations", self.base_url);
        let response = self.client.get(&url).send()?;

        if !response.status().is_success() {
            let status = response.status();
            return Err(format!("Failed to list conversations: HTTP {status}").into());
        }

        let list_response: ConversationListResponse = response.json()?;
        Ok(list_response.conversations)
    }

    /// Print a response in compact mode
    fn print_response(&self, resp: &ThinkResponse) {
        println!("Volt: {}", resp.text);
        if !self.debug_mode {
            let avg_gamma = if resp.gamma.is_empty() {
                0.0
            } else {
                resp.gamma.iter().sum::<f32>() / resp.gamma.len() as f32
            };
            println!(
                "     [γ: {:.2} | conv: {} | {}ms]",
                avg_gamma, resp.conversation_id, resp.timing_ms.total_ms as u32
            );
        } else {
            self.print_debug(resp);
        }
    }

    /// Print a response in debug mode
    fn print_debug(&self, resp: &ThinkResponse) {
        println!(
            "     [γ: {:.2} | conv: {} | iter: {} | {}ms]",
            resp.gamma.iter().sum::<f32>() / resp.gamma.len().max(1) as f32,
            resp.conversation_id,
            resp.iterations,
            resp.timing_ms.total_ms as u32
        );
        println!("     Timing: encode={:.1}ms, decode={:.1}ms",
            resp.timing_ms.encode_ms, resp.timing_ms.decode_ms);
        println!("     Memory: {} frames, {} ghosts", resp.memory_frame_count, resp.ghost_count);
        println!("     Safety score: {:.3}", resp.safety_score);

        if !resp.proof_steps.is_empty() {
            println!("     Proof chain:");
            for step in &resp.proof_steps {
                let status = if step.activated { "✓" } else { "✗" };
                println!(
                    "       → {} (sim: {:.2}, γ: {:.2}) {}",
                    step.strand_name, step.similarity, step.gamma_after, status
                );
            }
        }

        if !resp.slot_states.is_empty() {
            println!("     Slots: {} active", resp.slot_states.len());
            for slot in &resp.slot_states {
                println!(
                    "       [{}] {}: \"{}\" (γ={:.2}, source={})",
                    slot.index, slot.role, slot.word, slot.certainty, slot.source
                );
            }
        }
    }

    /// Handle a command (returns true if should exit)
    fn handle_command(&mut self, cmd: &str) -> Result<bool, Box<dyn Error>> {
        match cmd.trim() {
            "/debug" => {
                self.debug_mode = !self.debug_mode;
                println!(
                    "Debug mode: {}",
                    if self.debug_mode { "ON" } else { "OFF" }
                );
                Ok(false)
            }
            "/clear" => {
                self.conversation_id = None;
                println!("Started new conversation (will auto-create on first message)");
                Ok(false)
            }
            "/exit" | "/quit" => Ok(true),
            "/help" => {
                print_help();
                Ok(false)
            }
            "/list" => {
                let convs = self.list_conversations()?;
                if convs.is_empty() {
                    println!("No conversations yet.");
                } else {
                    println!("Conversations ({}):", convs.len());
                    for conv in convs {
                        let current = if Some(conv.id) == self.conversation_id {
                            " (current)"
                        } else {
                            ""
                        };
                        println!(
                            "  [{}] {} messages, last active: {}{}",
                            conv.id,
                            conv.message_count,
                            format_timestamp(conv.last_message_at),
                            current
                        );
                    }
                }
                Ok(false)
            }
            cmd if cmd.starts_with("/switch ") => {
                let id_str = cmd.trim_start_matches("/switch ").trim();
                match id_str.parse::<u64>() {
                    Ok(id) => {
                        self.conversation_id = Some(id);
                        println!("Switched to conversation {}", id);
                        Ok(false)
                    }
                    Err(_) => {
                        println!("Invalid conversation ID: {}", id_str);
                        Ok(false)
                    }
                }
            }
            _ => {
                println!("Unknown command: {}", cmd);
                println!("Type /help for available commands");
                Ok(false)
            }
        }
    }
}

/// Format a microsecond timestamp for display
fn format_timestamp(micros: u64) -> String {
    let secs = micros / 1_000_000;
    let datetime = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(secs);
    let now = std::time::SystemTime::now();

    if let Ok(elapsed) = now.duration_since(datetime) {
        let mins = elapsed.as_secs() / 60;
        if mins < 60 {
            format!("{}m ago", mins)
        } else {
            let hours = mins / 60;
            if hours < 24 {
                format!("{}h ago", hours)
            } else {
                format!("{}d ago", hours / 24)
            }
        }
    } else {
        "just now".to_string()
    }
}

/// Print help text
fn print_help() {
    println!("Volt X Chat Commands:");
    println!("  /help          Show this help message");
    println!("  /debug         Toggle debug mode (show proof chain, slots)");
    println!("  /clear         Start a new conversation");
    println!("  /list          List all conversations");
    println!("  /switch <id>   Switch to a different conversation");
    println!("  /exit, /quit   Exit the chat");
    println!();
    println!("Just type a message to chat with Volt!");
}

fn main() -> Result<(), Box<dyn Error>> {
    // Parse command-line arguments
    let args: Vec<String> = std::env::args().collect();
    let mut url = "http://localhost:8080".to_string();
    let mut initial_conversation: Option<u64> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--url" => {
                if i + 1 < args.len() {
                    url = args[i + 1].clone();
                    i += 2;
                } else {
                    eprintln!("Error: --url requires a value");
                    std::process::exit(1);
                }
            }
            "--conversation" => {
                if i + 1 < args.len() {
                    initial_conversation = args[i + 1].parse().ok();
                    i += 2;
                } else {
                    eprintln!("Error: --conversation requires a value");
                    std::process::exit(1);
                }
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                eprintln!("Usage: volt-chat [--url <server-url>] [--conversation <id>]");
                std::process::exit(1);
            }
        }
    }

    let mut client = ChatClient::new(url);
    client.conversation_id = initial_conversation;

    let mut rl = Editor::<(), _>::new()?;

    println!("Volt X Chat (v{})", env!("CARGO_PKG_VERSION"));
    println!("Connected to: {}", client.base_url);
    println!("Type /help for commands, /exit to quit");
    println!();

    loop {
        let readline = rl.readline("You: ");
        match readline {
            Ok(line) => {
                if line.trim().is_empty() {
                    continue;
                }

                rl.add_history_entry(&line)?;

                if line.starts_with('/') {
                    match client.handle_command(&line) {
                        Ok(should_exit) => {
                            if should_exit {
                                break;
                            }
                        }
                        Err(e) => eprintln!("Command error: {}", e),
                    }
                    continue;
                }

                match client.send_message(&line) {
                    Ok(resp) => client.print_response(&resp),
                    Err(e) => eprintln!("Error: {}", e),
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C");
                println!("Type /exit to quit");
            }
            Err(ReadlineError::Eof) => {
                println!("Goodbye!");
                break;
            }
            Err(err) => {
                eprintln!("Error: {}", err);
                break;
            }
        }
    }

    Ok(())
}
