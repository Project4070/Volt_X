class VoltChat {
    constructor() {
        this.currentConversation = null;
        this.debugMode = false;
        this.streamingMode = true; // Default to streaming
        this.conversations = [];
        this.init();
    }

    async init() {
        this.setupEventListeners();
        await this.loadConversations();

        // Auto-create first conversation if none exist
        if (this.conversations.length === 0) {
            await this.createConversation();
        } else {
            // Select the most recent conversation
            this.switchConversation(this.conversations[0].id);
        }
    }

    setupEventListeners() {
        const sendBtn = document.getElementById('send-btn');
        const input = document.getElementById('input');
        const newConvBtn = document.getElementById('new-conv-btn');
        const debugToggle = document.getElementById('debug-toggle');
        const streamToggle = document.getElementById('stream-toggle');

        sendBtn.addEventListener('click', () => this.sendMessage());

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize textarea
        input.addEventListener('input', () => {
            input.style.height = 'auto';
            input.style.height = Math.min(input.scrollHeight, 150) + 'px';
        });

        newConvBtn.addEventListener('click', () => this.createConversation());
        debugToggle.addEventListener('click', () => this.toggleDebug());
        streamToggle.addEventListener('click', () => this.toggleStreaming());
    }

    async loadConversations() {
        try {
            const response = await fetch('/api/conversations');
            if (!response.ok) throw new Error('Failed to load conversations');

            const data = await response.json();
            this.conversations = data.conversations;
            this.renderConversationList();
        } catch (error) {
            console.error('Error loading conversations:', error);
            this.showError('Failed to load conversations');
        }
    }

    renderConversationList() {
        const list = document.getElementById('conv-list');
        list.innerHTML = '';

        this.conversations.forEach(conv => {
            const item = document.createElement('div');
            item.className = 'conversation-item';
            if (conv.id === this.currentConversation) {
                item.classList.add('active');
            }

            const title = document.createElement('div');
            title.className = 'conv-item-title';
            title.textContent = `Conversation ${conv.id.toString().slice(-6)}`;

            const meta = document.createElement('div');
            meta.className = 'conv-item-meta';
            meta.textContent = `${conv.message_count} messages • ${this.formatTime(conv.last_message_at)}`;

            item.appendChild(title);
            item.appendChild(meta);
            item.addEventListener('click', () => this.switchConversation(conv.id));
            list.appendChild(item);
        });
    }

    async createConversation() {
        try {
            const response = await fetch('/api/conversations', {
                method: 'POST'
            });
            if (!response.ok) throw new Error('Failed to create conversation');

            const data = await response.json();
            const newConv = {
                id: data.conversation_id,
                created_at: Date.now() * 1000,
                last_message_at: Date.now() * 1000,
                message_count: 0
            };

            this.conversations.unshift(newConv);
            this.switchConversation(newConv.id);
            this.renderConversationList();
        } catch (error) {
            console.error('Error creating conversation:', error);
            this.showError('Failed to create conversation');
        }
    }

    async switchConversation(id) {
        this.currentConversation = id;
        this.renderConversationList();
        await this.loadConversationHistory(id);
        this.updateConversationHeader();
    }

    async loadConversationHistory(id) {
        const messagesContainer = document.getElementById('messages');
        messagesContainer.innerHTML = '';

        try {
            const response = await fetch(`/api/conversations/${id}/history`);
            if (!response.ok) {
                // If 404, conversation is new/empty
                if (response.status === 404) return;
                throw new Error('Failed to load history');
            }

            const data = await response.json();
            data.messages.forEach((msg, index) => {
                // Alternate between user and assistant messages
                const role = index % 2 === 0 ? 'user' : 'assistant';
                this.appendMessageToDOM(role, msg.text, {
                    gamma: msg.gamma,
                    timestamp: msg.timestamp
                });
            });

            this.scrollToBottom();
        } catch (error) {
            console.error('Error loading history:', error);
        }
    }

    updateConversationHeader() {
        const conv = this.conversations.find(c => c.id === this.currentConversation);
        if (conv) {
            document.getElementById('conv-title').textContent = `Conversation ${conv.id.toString().slice(-6)}`;
            document.getElementById('conv-stats').textContent = `${conv.message_count} messages`;
        }
    }

    async sendMessage() {
        if (this.streamingMode) {
            await this.sendMessageStream();
        } else {
            await this.sendMessageNonStream();
        }
    }

    async sendMessageNonStream() {
        const input = document.getElementById('input');
        const text = input.value.trim();

        if (!text) return;

        // Clear input immediately
        input.value = '';
        input.style.height = 'auto';

        // Disable send button
        const sendBtn = document.getElementById('send-btn');
        sendBtn.disabled = true;

        // Append user message
        this.appendMessageToDOM('user', text);

        try {
            const response = await fetch('/api/think', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    conversation_id: this.currentConversation
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Request failed');
            }

            const data = await response.json();

            // Update conversation_id if it was auto-created
            if (!this.currentConversation) {
                this.currentConversation = data.conversation_id;
                await this.loadConversations();
            }

            // Append assistant message
            this.appendMessageToDOM('assistant', data.text, data);

            // Update conversation metadata
            const conv = this.conversations.find(c => c.id === this.currentConversation);
            if (conv) {
                conv.message_count += 2; // User + assistant
                conv.last_message_at = Date.now() * 1000;
                this.updateConversationHeader();
                this.renderConversationList();
            }

            this.scrollToBottom();
        } catch (error) {
            console.error('Error sending message:', error);
            this.showError(error.message);
        } finally {
            sendBtn.disabled = false;
            input.focus();
        }
    }

    async sendMessageStream() {
        const input = document.getElementById('input');
        const text = input.value.trim();

        if (!text) return;

        // Clear input immediately
        input.value = '';
        input.style.height = 'auto';

        // Disable send button
        const sendBtn = document.getElementById('send-btn');
        sendBtn.disabled = true;

        // Append user message
        this.appendMessageToDOM('user', text);

        // Create placeholder for assistant response
        const messagesContainer = document.getElementById('messages');
        const assistantMessage = document.createElement('div');
        assistantMessage.className = 'message assistant';

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.textContent = 'Preparing...';
        assistantMessage.appendChild(bubble);

        const meta = document.createElement('div');
        meta.className = 'message-meta';
        assistantMessage.appendChild(meta);

        messagesContainer.appendChild(assistantMessage);
        this.scrollToBottom();

        try {
            const response = await fetch('/api/think/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    conversation_id: this.currentConversation
                })
            });

            if (!response.ok) {
                throw new Error('Stream request failed');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n\n');
                buffer = lines.pop() || ''; // Keep incomplete line in buffer

                for (const line of lines) {
                    if (!line.trim() || !line.startsWith('data: ')) continue;

                    const dataStr = line.substring(6); // Remove "data: " prefix
                    try {
                        const event = JSON.parse(dataStr);

                        if (event.type === 'status') {
                            bubble.textContent = event.data;
                        } else if (event.type === 'encoding') {
                            bubble.textContent = 'Encoding...';
                        } else if (event.type === 'thinking') {
                            bubble.textContent = 'Thinking...';
                        } else if (event.type === 'complete') {
                            const data = event.data;

                            // Update conversation_id if it was auto-created
                            if (!this.currentConversation) {
                                this.currentConversation = data.conversation_id;
                                await this.loadConversations();
                            }

                            // Update message content
                            bubble.textContent = data.text;

                            // Update metadata
                            const avgGamma = data.gamma.reduce((a, b) => a + b, 0) / data.gamma.length;
                            meta.textContent = `γ: ${avgGamma.toFixed(2)}`;
                            if (data.timing_ms) {
                                meta.textContent += ` • ${Math.round(data.timing_ms.total_ms)}ms`;
                            }

                            // Add debug info if enabled
                            if (this.debugMode && data.slot_states) {
                                const debugInfo = this.renderDebugInfo(data);
                                assistantMessage.appendChild(debugInfo);
                            }

                            // Update conversation metadata
                            const conv = this.conversations.find(c => c.id === this.currentConversation);
                            if (conv) {
                                conv.message_count += 2;
                                conv.last_message_at = Date.now() * 1000;
                                this.updateConversationHeader();
                                this.renderConversationList();
                            }
                        } else if (event.type === 'error') {
                            bubble.textContent = `Error: ${event.data}`;
                            bubble.style.color = '#ff4444';
                        }

                        this.scrollToBottom();
                    } catch (e) {
                        console.error('Failed to parse SSE event:', e, dataStr);
                    }
                }
            }
        } catch (error) {
            console.error('Error in stream:', error);
            bubble.textContent = `Error: ${error.message}`;
            bubble.style.color = '#ff4444';
        } finally {
            sendBtn.disabled = false;
            input.focus();
        }
    }

    appendMessageToDOM(role, text, metadata = null) {
        const messagesContainer = document.getElementById('messages');
        const message = document.createElement('div');
        message.className = `message ${role}`;

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.textContent = text;
        message.appendChild(bubble);

        if (metadata) {
            const meta = document.createElement('div');
            meta.className = 'message-meta';

            if (role === 'assistant' && metadata.gamma) {
                const avgGamma = metadata.gamma.reduce((a, b) => a + b, 0) / metadata.gamma.length;
                meta.textContent = `γ: ${avgGamma.toFixed(2)}`;

                if (metadata.timing_ms) {
                    meta.textContent += ` • ${Math.round(metadata.timing_ms.total_ms)}ms`;
                }
            }
            message.appendChild(meta);

            // Add debug info if debug mode is on
            if (this.debugMode && role === 'assistant' && metadata.slot_states) {
                const debugInfo = this.renderDebugInfo(metadata);
                message.appendChild(debugInfo);
            }
        }

        messagesContainer.appendChild(message);
    }

    renderDebugInfo(data) {
        const debugDiv = document.createElement('div');
        debugDiv.className = 'debug-info';

        const info = [];
        info.push(`Iterations: ${data.iterations}`);
        info.push(`Memory: ${data.memory_frame_count} frames, ${data.ghost_count} ghosts`);
        info.push(`Safety: ${data.safety_score.toFixed(3)}`);
        info.push(`Timing: encode=${data.timing_ms.encode_ms.toFixed(1)}ms, decode=${data.timing_ms.decode_ms.toFixed(1)}ms`);

        debugDiv.innerHTML = info.map(line => `<div class="debug-info-row">${this.escapeHtml(line)}</div>`).join('');

        if (data.proof_steps && data.proof_steps.length > 0) {
            const proofChain = document.createElement('div');
            proofChain.className = 'proof-chain';
            proofChain.innerHTML = '<strong>Proof Chain:</strong>';

            data.proof_steps.forEach(step => {
                const stepDiv = document.createElement('div');
                stepDiv.className = 'proof-step';
                const status = step.activated ? '✓' : '✗';
                stepDiv.textContent = `→ ${step.strand_name} (sim: ${step.similarity.toFixed(2)}, γ: ${step.gamma_after.toFixed(2)}) ${status}`;
                proofChain.appendChild(stepDiv);
            });

            debugDiv.appendChild(proofChain);
        }

        if (data.slot_states && data.slot_states.length > 0) {
            const slots = document.createElement('div');
            slots.className = 'proof-chain';
            slots.innerHTML = `<strong>Slots (${data.slot_states.length} active):</strong>`;

            data.slot_states.slice(0, 5).forEach(slot => {
                const slotDiv = document.createElement('div');
                slotDiv.className = 'proof-step';
                slotDiv.textContent = `[${slot.index}] ${slot.role}: "${slot.word}" (γ=${slot.certainty.toFixed(2)}, ${slot.source})`;
                slots.appendChild(slotDiv);
            });

            debugDiv.appendChild(slots);
        }

        return debugDiv;
    }

    toggleDebug() {
        this.debugMode = !this.debugMode;
        const toggle = document.getElementById('debug-toggle');
        toggle.classList.toggle('active');

        // Re-render messages to show/hide debug info
        if (this.currentConversation) {
            this.loadConversationHistory(this.currentConversation);
        }
    }

    toggleStreaming() {
        this.streamingMode = !this.streamingMode;
        const toggle = document.getElementById('stream-toggle');
        toggle.classList.toggle('active');
        toggle.textContent = this.streamingMode ? '⚡ Streaming' : '⚡ Batch';
    }

    showError(message) {
        const messagesContainer = document.getElementById('messages');
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = `Error: ${message}`;
        messagesContainer.appendChild(errorDiv);
        this.scrollToBottom();
    }

    scrollToBottom() {
        const container = document.getElementById('chat-container');
        container.scrollTop = container.scrollHeight;
    }

    formatTime(micros) {
        const now = Date.now() * 1000;
        const diff = now - micros;
        const seconds = Math.floor(diff / 1_000_000);

        if (seconds < 60) return 'just now';
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
        if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
        return `${Math.floor(seconds / 86400)}d ago`;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize the app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new VoltChat();
});
