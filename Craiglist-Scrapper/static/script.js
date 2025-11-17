// API Base URL - Update this to match your server URL
const API_BASE_URL = window.location.origin;

// Global state
let currentThreadId = null;
let currentTab = 'all';
let allMessages = [];
let allThreads = [];
let filteredThreads = [];

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    // Animate header on load
    gsap.from('header', {
        duration: 0.8,
        y: -30,
        opacity: 0,
        ease: 'power3.out'
    });
    
    // Animate sidebar
    gsap.from('.sidebar', {
        duration: 0.8,
        x: -30,
        opacity: 0,
        delay: 0.2,
        ease: 'power3.out'
    });
    
    // Animate chat area
    gsap.from('.chat-area', {
        duration: 0.8,
        x: 30,
        opacity: 0,
        delay: 0.4,
        ease: 'power3.out'
    });
    
    loadConversations();
    
    // Event listeners with interactive effects
    const refreshBtn = document.getElementById('refreshBtn');
    refreshBtn.addEventListener('click', (e) => {
        // Create ripple effect
        createRippleEffect(e, refreshBtn);
        
        // Animate refresh button
        if (typeof gsap !== 'undefined') {
            gsap.to('.refresh-icon', {
                rotation: 360,
                duration: 0.5,
                ease: 'power2.inOut'
            });
        }
        loadConversations();
    });
    
    // Search functionality
    const searchInput = document.getElementById('searchInput');
    searchInput.addEventListener('input', (e) => {
        filterThreads(e.target.value);
    });
    
    // Tab switching with interactive effects
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            // Create ripple effect
            createRippleEffect(e, btn);
            
            currentTab = e.target.dataset.tab;
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            
            // Animate tab switch
            if (typeof gsap !== 'undefined') {
                gsap.from(e.target, {
                    scale: 0.9,
                    duration: 0.2,
                    ease: 'back.out(1.7)'
                });
            }
            
            displayMessages();
        });
    });
    
    // Add interactive effects to thread items
    document.addEventListener('click', (e) => {
        const threadItem = e.target.closest('.thread-item');
        if (threadItem) {
            createRippleEffect(e, threadItem);
        }
    });
});

/**
 * Create iOS-style ripple effect on button click
 */
// Debounce ripple effects
let rippleTimeout = null;
function createRippleEffect(event, element) {
    // Skip ripple on rapid clicks
    if (element.dataset.rippleActive === 'true') return;
    
    // Clear any pending ripple
    if (rippleTimeout) {
        clearTimeout(rippleTimeout);
    }
    
    element.dataset.rippleActive = 'true';
    
    const ripple = document.createElement('span');
    const rect = element.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const x = event.clientX - rect.left - size / 2;
    const y = event.clientY - rect.top - size / 2;
    
    ripple.style.width = ripple.style.height = size + 'px';
    ripple.style.left = x + 'px';
    ripple.style.top = y + 'px';
    ripple.classList.add('ripple');
    
    if (!element.style.position || element.style.position === 'static') {
        element.style.position = 'relative';
    }
    if (!element.style.overflow || element.style.overflow === 'visible') {
        element.style.overflow = 'hidden';
    }
    element.appendChild(ripple);
    
    // Faster, simpler animation
    requestAnimationFrame(() => {
        ripple.style.animation = 'ripple-animation 0.3s ease-out';
        
        rippleTimeout = setTimeout(() => {
            ripple.remove();
            element.dataset.rippleActive = 'false';
        }, 300);
    });
}

/**
 * Filter threads by phone number
 */
function filterThreads(searchTerm) {
    if (!searchTerm.trim()) {
        filteredThreads = allThreads;
    } else {
        const searchLower = searchTerm.toLowerCase().replace(/\D/g, '');
        filteredThreads = allThreads.filter(thread => {
            const phoneClean = thread.phone_number.replace(/\D/g, '').toLowerCase();
            return phoneClean.includes(searchLower);
        });
    }
    
    displayThreadList(filteredThreads);
}

/**
 * Load all conversation threads
 */
async function loadConversations() {
    const threadList = document.getElementById('threadList');
    threadList.innerHTML = '<div class="loading">Loading conversations...</div>';
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/conversations`);
        const data = await response.json();
        
        console.log('API Response:', data);
        
        if (data.success) {
            if (data.threads && data.threads.length > 0) {
                allThreads = data.threads;
                filteredThreads = data.threads;
                displayThreadList(data.threads);
            } else {
                threadList.innerHTML = '<div class="empty-state">No conversations yet. Send a message to start a conversation.</div>';
            }
        } else {
            threadList.innerHTML = `<div class="error">Error: ${data.error || 'Unknown error'}</div>`;
        }
    } catch (error) {
        console.error('Error loading conversations:', error);
        threadList.innerHTML = `<div class="error">Error loading conversations: ${error.message}. Please check the console for details.</div>`;
    }
}

/**
 * Display list of conversation threads
 */
function displayThreadList(threads) {
    const threadList = document.getElementById('threadList');
    threadList.innerHTML = '';
    
    if (threads.length === 0) {
        threadList.innerHTML = '<div class="empty-state">No conversations found</div>';
        return;
    }
    
    threads.forEach((thread, index) => {
        const threadItem = document.createElement('div');
        threadItem.className = 'thread-item';
        threadItem.dataset.threadId = thread.thread_id;
        
        const lastMessage = new Date(thread.last_message).toLocaleDateString();
        
        threadItem.innerHTML = `
            <div class="thread-phone">${formatPhoneNumber(thread.phone_number)}</div>
            <div class="thread-meta">
                <span>${thread.message_count} messages</span>
                <span>${lastMessage}</span>
            </div>
            <div class="thread-stats">
                <span class="thread-stat inbound">📥 ${thread.inbound_count}</span>
                <span class="thread-stat outbound">📤 ${thread.outbound_count}</span>
            </div>
        `;
        
        threadItem.addEventListener('click', () => {
            selectThread(thread.thread_id, thread.phone_number);
        });
        
        threadList.appendChild(threadItem);
        
        // Animate thread items with stagger
        gsap.from(threadItem, {
            opacity: 0,
            x: -20,
            duration: 0.4,
            delay: index * 0.05,
            ease: 'power2.out'
        });
    });
}

/**
 * Select a thread and load its messages
 */
async function selectThread(threadId, phoneNumber) {
    currentThreadId = threadId;
    
    // Update active thread in sidebar with animation
    document.querySelectorAll('.thread-item').forEach(item => {
        if (item.dataset.threadId === threadId) {
            item.classList.add('active');
            gsap.to(item, {
                scale: 1.02,
                duration: 0.2,
                yoyo: true,
                repeat: 1
            });
        } else {
            item.classList.remove('active');
        }
    });
    
    // Animate transition
    const emptyState = document.getElementById('emptyState');
    const chatContainer = document.getElementById('chatContainer');
    
    if (emptyState.style.display !== 'none') {
        gsap.to(emptyState, {
            opacity: 0,
            y: -20,
            duration: 0.3,
            onComplete: () => {
                emptyState.style.display = 'none';
                chatContainer.style.display = 'flex';
                gsap.from(chatContainer, {
                    opacity: 0,
                    y: 20,
                    duration: 0.4,
                    ease: 'power2.out'
                });
            }
        });
    }
    
    // Update chat header
    document.getElementById('chatTitle').textContent = formatPhoneNumber(phoneNumber);
    
    // Load messages
    await loadMessages(threadId);
}

/**
 * Load messages for a specific thread
 */
async function loadMessages(threadId) {
    const messagesContainer = document.getElementById('messagesContainer');
    messagesContainer.innerHTML = '<div class="loading">Loading messages...</div>';
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/conversations/${threadId}`);
        const data = await response.json();
        
        console.log('Messages API Response:', data); // Debug log
        
        if (data.success) {
            allMessages = data.messages || [];
            console.log(`Loaded ${allMessages.length} messages for thread ${threadId}`); // Debug log
            
            if (allMessages.length > 0) {
                document.getElementById('chatMeta').textContent = 
                    `Phone: ${formatPhoneNumber(allMessages[0]?.phone_number || 'N/A')} | Messages: ${data.count || allMessages.length}`;
                displayMessages();
            } else {
                messagesContainer.innerHTML = '<div class="empty-state">No messages found in this conversation</div>';
            }
        } else {
            console.error('API returned error:', data.error);
            messagesContainer.innerHTML = `<div class="error">Error loading messages: ${data.error || 'Unknown error'}</div>`;
        }
    } catch (error) {
        console.error('Error loading messages:', error);
        messagesContainer.innerHTML = '<div class="error">Error loading messages. Please try again.</div>';
    }
}

/**
 * Display messages based on current tab filter
 */
function displayMessages() {
    const messagesContainer = document.getElementById('messagesContainer');
    messagesContainer.innerHTML = '';
    
    if (!allMessages || allMessages.length === 0) {
        messagesContainer.innerHTML = '<div class="empty-state">No messages available</div>';
        return;
    }
    
    let filteredMessages = allMessages;
    
    if (currentTab === 'inbound') {
        filteredMessages = allMessages.filter(msg => msg.message_type === 'inbound');
    } else if (currentTab === 'outbound') {
        filteredMessages = allMessages.filter(msg => msg.message_type === 'outbound');
    }
    
    console.log(`Displaying ${filteredMessages.length} messages (tab: ${currentTab})`); // Debug log
    
    if (filteredMessages.length === 0) {
        messagesContainer.innerHTML = '<div class="empty-state">No messages in this category</div>';
        return;
    }
    
    // Batch DOM updates for better performance
    const fragment = document.createDocumentFragment();
    
    filteredMessages.forEach((message, index) => {
        const messageDiv = createMessageElement(message);
        fragment.appendChild(messageDiv);
    });
    
    messagesContainer.appendChild(fragment);
    
    // Simplified animation - only animate first few messages for performance
    if (typeof gsap !== 'undefined') {
        const messageElements = messagesContainer.querySelectorAll('.message');
        const maxAnimate = Math.min(20, messageElements.length); // Only animate first 20
        
        if (maxAnimate > 0) {
            gsap.fromTo(Array.from(messageElements).slice(0, maxAnimate), 
                { opacity: 0 },
                {
                    opacity: 1,
                    duration: 0.2,
                    stagger: 0.01,
                    ease: 'none'
                }
            );
        }
        
        // Show remaining messages instantly
        if (messageElements.length > maxAnimate) {
            Array.from(messageElements).slice(maxAnimate).forEach(el => {
                el.style.opacity = '1';
            });
        }
    }
    
    // Smooth scroll to bottom (use native for better performance)
    requestAnimationFrame(() => {
        messagesContainer.scrollTo({
            top: messagesContainer.scrollHeight,
            behavior: 'smooth'
        });
    });
}

/**
 * Create a message element
 */
function createMessageElement(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${message.message_type}`;
    
    const isInbound = message.message_type === 'inbound';
    const roleLabel = isInbound ? 'Customer' : 'AI Assistant';
    const timestamp = new Date(message.created_at).toLocaleString();
    
    messageDiv.innerHTML = `
        <div class="message-bubble">
            <div class="message-content">${escapeHtml(message.content)}</div>
            <div class="message-meta">
                <span>${roleLabel}</span>
                <span class="message-time">${timestamp}</span>
            </div>
        </div>
    `;
    
    return messageDiv;
}

/**
 * Format phone number for display
 */
function formatPhoneNumber(phone) {
    if (!phone) return 'N/A';
    // Format: +1 (XXX) XXX-XXXX
    const cleaned = phone.replace(/\D/g, '');
    if (cleaned.length === 11 && cleaned.startsWith('1')) {
        return `+1 (${cleaned.slice(1, 4)}) ${cleaned.slice(4, 7)}-${cleaned.slice(7)}`;
    } else if (cleaned.length === 10) {
        return `(${cleaned.slice(0, 3)}) ${cleaned.slice(3, 6)}-${cleaned.slice(6)}`;
    }
    return phone;
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
