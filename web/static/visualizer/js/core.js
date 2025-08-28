/**
 * 🧠 MeRNSTA Memory Graph Visualizer - Core JavaScript
 * Phase 34 - Core functionality and utilities
 */

// === GLOBAL CONFIGURATION ===
window.MeRNSTA = window.MeRNSTA || {};

MeRNSTA.Core = {
    // Configuration
    config: null,
    apiBaseUrl: 'http://127.0.0.1:8181',
    
    // State management
    connectionStatus: 'disconnected',
    updateCount: 0,
    lastUpdate: null,
    autoRefresh: true,
    refreshInterval: null,
    
    // Event handlers
    eventHandlers: {},
    
    /**
     * Initialize the core system
     */
    init: function(config = {}) {
        this.config = config;
        if (config.api_base_url) {
            this.apiBaseUrl = config.api_base_url;
        }
        
        console.log('🧠 MeRNSTA Visualizer Core initialized');
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Check connection
        this.checkConnection();
        
        // Start auto-refresh if enabled
        if (this.config.real_time_updates !== false) {
            this.startAutoRefresh();
        }
        
        return this;
    },
    
    /**
     * Set up global event listeners
     */
    setupEventListeners: function() {
        // Theme toggle
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', this.toggleTheme.bind(this));
        }
        
        // Connection status monitoring
        window.addEventListener('online', () => this.updateConnectionStatus('online'));
        window.addEventListener('offline', () => this.updateConnectionStatus('offline'));
        
        // Auto-refresh controls
        const autoRefreshToggle = document.getElementById('auto-refresh');
        if (autoRefreshToggle) {
            autoRefreshToggle.addEventListener('change', (e) => {
                this.autoRefresh = e.target.checked;
                if (this.autoRefresh) {
                    this.startAutoRefresh();
                } else {
                    this.stopAutoRefresh();
                }
            });
        }
        
        // Global error handler
        window.addEventListener('error', this.handleGlobalError.bind(this));
    },
    
    /**
     * API request wrapper
     */
    async apiRequest(endpoint, options = {}) {
        const url = `${this.apiBaseUrl}${endpoint}`;
        const defaultOptions = {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        };
        
        const requestOptions = { ...defaultOptions, ...options };
        
        try {
            console.log(`📡 API Request: ${requestOptions.method} ${url}`);
            const response = await fetch(url, requestOptions);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            this.updateConnectionStatus('connected');
            return { success: true, data, response };
            
        } catch (error) {
            console.error('❌ API Request failed:', error);
            this.updateConnectionStatus('error');
            return { success: false, error: error.message, response: null };
        }
    },
    
    /**
     * Get visualizer data for a specific module
     */
    async getVisualizerData(module, timeRange = '24h', limit = 100, filters = null) {
        const requestData = {
            module: module,
            time_range: timeRange,
            limit: limit,
            filters: filters
        };
        
        return await this.apiRequest('/visualizer/data', {
            method: 'POST',
            body: JSON.stringify(requestData)
        });
    },
    
    /**
     * Get real-time cognitive events
     */
    async getCognitiveEvents() {
        return await this.apiRequest('/visualizer/events');
    },
    
    /**
     * Get system status
     */
    async getSystemStatus() {
        return await this.apiRequest('/status');
    },
    
    /**
     * Update connection status indicator
     */
    updateConnectionStatus: function(status) {
        this.connectionStatus = status;
        
        const indicators = document.querySelectorAll('.status-dot');
        const statusTexts = document.querySelectorAll('.status-text');
        
        indicators.forEach(dot => {
            dot.className = 'status-dot';
            if (status === 'connected') {
                dot.classList.add('connected');
            } else if (status === 'loading') {
                dot.classList.add('loading');
            }
        });
        
        statusTexts.forEach(text => {
            switch (status) {
                case 'connected':
                    text.textContent = 'Connected';
                    break;
                case 'loading':
                    text.textContent = 'Loading...';
                    break;
                case 'error':
                    text.textContent = 'Error';
                    break;
                default:
                    text.textContent = 'Disconnected';
            }
        });
        
        // Emit status change event
        this.emit('connectionStatusChanged', { status, timestamp: new Date() });
    },
    
    /**
     * Check API connection
     */
    async checkConnection() {
        this.updateConnectionStatus('loading');
        const result = await this.getSystemStatus();
        
        if (result.success) {
            this.updateConnectionStatus('connected');
            this.updateLastUpdate();
        } else {
            this.updateConnectionStatus('error');
        }
        
        return result.success;
    },
    
    /**
     * Start auto-refresh
     */
    startAutoRefresh: function() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
        
        const interval = (this.config.update_interval || 5) * 1000;
        this.refreshInterval = setInterval(() => {
            if (this.autoRefresh) {
                this.emit('autoRefresh', { timestamp: new Date() });
            }
        }, interval);
        
        console.log(`🔄 Auto-refresh started (${interval/1000}s interval)`);
    },
    
    /**
     * Stop auto-refresh
     */
    stopAutoRefresh: function() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
        console.log('⏸️ Auto-refresh stopped');
    },
    
    /**
     * Update last update timestamp
     */
    updateLastUpdate: function() {
        this.lastUpdate = new Date();
        this.updateCount++;
        
        const updateElements = document.querySelectorAll('#last-update');
        const timestamp = this.formatTimestamp(this.lastUpdate);
        
        updateElements.forEach(el => {
            el.textContent = timestamp;
        });
        
        const counterElements = document.querySelectorAll('#update-counter');
        counterElements.forEach(el => {
            el.textContent = `Updates: ${this.updateCount}`;
        });
    },
    
    /**
     * Format timestamp for display
     */
    formatTimestamp: function(date) {
        if (!date) return '--';
        
        const now = new Date();
        const diff = now - date;
        
        if (diff < 60000) { // Less than 1 minute
            return 'Just now';
        } else if (diff < 3600000) { // Less than 1 hour
            const minutes = Math.floor(diff / 60000);
            return `${minutes}m ago`;
        } else if (diff < 86400000) { // Less than 1 day
            const hours = Math.floor(diff / 3600000);
            return `${hours}h ago`;
        } else {
            return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
        }
    },
    
    /**
     * Toggle theme between light and dark
     */
    toggleTheme: function() {
        const body = document.body;
        const isLight = body.classList.contains('theme-light');
        
        if (isLight) {
            body.classList.remove('theme-light');
            body.classList.add('theme-dark');
        } else {
            body.classList.remove('theme-dark');
            body.classList.add('theme-light');
        }
        
        // Update theme toggle button
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.textContent = isLight ? '☀️' : '🌙';
        }
        
        // Save theme preference
        localStorage.setItem('mernsta-theme', isLight ? 'dark' : 'light');
        
        this.emit('themeChanged', { theme: isLight ? 'dark' : 'light' });
    },
    
    /**
     * Show loading state
     */
    showLoading: function(element) {
        if (typeof element === 'string') {
            element = document.getElementById(element);
        }
        
        if (element) {
            const overlay = element.querySelector('.loading-overlay');
            if (overlay) {
                overlay.style.display = 'flex';
            }
        }
    },
    
    /**
     * Hide loading state
     */
    hideLoading: function(element) {
        if (typeof element === 'string') {
            element = document.getElementById(element);
        }
        
        if (element) {
            const overlay = element.querySelector('.loading-overlay');
            if (overlay) {
                overlay.style.display = 'none';
            }
        }
    },
    
    /**
     * Show error state
     */
    showError: function(element, message) {
        if (typeof element === 'string') {
            element = document.getElementById(element);
        }
        
        if (element) {
            const overlay = element.querySelector('.error-overlay');
            const messageEl = element.querySelector('#error-message');
            
            if (overlay && messageEl) {
                messageEl.textContent = message || 'An error occurred';
                overlay.style.display = 'flex';
            }
        }
    },
    
    /**
     * Hide error state
     */
    hideError: function(element) {
        if (typeof element === 'string') {
            element = document.getElementById(element);
        }
        
        if (element) {
            const overlay = element.querySelector('.error-overlay');
            if (overlay) {
                overlay.style.display = 'none';
            }
        }
    },
    
    /**
     * Event system
     */
    on: function(event, handler) {
        if (!this.eventHandlers[event]) {
            this.eventHandlers[event] = [];
        }
        this.eventHandlers[event].push(handler);
    },
    
    off: function(event, handler) {
        if (this.eventHandlers[event]) {
            const index = this.eventHandlers[event].indexOf(handler);
            if (index > -1) {
                this.eventHandlers[event].splice(index, 1);
            }
        }
    },
    
    emit: function(event, data) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`Error in event handler for '${event}':`, error);
                }
            });
        }
    },
    
    /**
     * Handle global errors
     */
    handleGlobalError: function(event) {
        console.error('Global error:', event.error);
        this.emit('globalError', { error: event.error, event });
    },
    
    /**
     * Utility functions
     */
    utils: {
        /**
         * Debounce function calls
         */
        debounce: function(func, wait) {
            let timeout;
            return function executedFunction(...args) {
                const later = () => {
                    clearTimeout(timeout);
                    func(...args);
                };
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            };
        },
        
        /**
         * Throttle function calls
         */
        throttle: function(func, limit) {
            let inThrottle;
            return function() {
                const args = arguments;
                const context = this;
                if (!inThrottle) {
                    func.apply(context, args);
                    inThrottle = true;
                    setTimeout(() => inThrottle = false, limit);
                }
            };
        },
        
        /**
         * Generate random ID
         */
        generateId: function(prefix = 'id') {
            return `${prefix}_${Math.random().toString(36).substr(2, 9)}`;
        },
        
        /**
         * Deep clone object
         */
        deepClone: function(obj) {
            return JSON.parse(JSON.stringify(obj));
        },
        
        /**
         * Format numbers for display
         */
        formatNumber: function(num, decimals = 2) {
            if (isNaN(num)) return '--';
            if (num === 0) return '0';
            
            if (Math.abs(num) < 0.01) {
                return num.toExponential(1);
            }
            
            return num.toFixed(decimals);
        }
    }
};

// === GLOBAL FUNCTIONS ===

/**
 * Navigate functions
 */
function openModule(moduleName) {
    window.location.href = `/visualizer/module/${moduleName}`;
}

function openEventsFullView() {
    window.location.href = '/visualizer/events';
}

function goBack() {
    if (document.referrer && document.referrer.includes('/visualizer')) {
        history.back();
    } else {
        window.location.href = '/visualizer/';
    }
}

/**
 * Module control functions
 */
function resetView() {
    MeRNSTA.Core.emit('resetView');
}

function exportData() {
    MeRNSTA.Core.emit('exportData');
}

function retryLoad() {
    MeRNSTA.Core.emit('retryLoad');
}

// === INITIALIZATION ===
document.addEventListener('DOMContentLoaded', function() {
    // Load theme preference
    const savedTheme = localStorage.getItem('mernsta-theme');
    if (savedTheme) {
        document.body.classList.add(`theme-${savedTheme}`);
    }
    
    // Initialize core if config is available
    if (window.VISUALIZER_CONFIG) {
        MeRNSTA.Core.init(window.VISUALIZER_CONFIG);
    } else if (window.MODULE_CONFIG) {
        MeRNSTA.Core.init(window.MODULE_CONFIG);
    } else if (window.EVENTS_CONFIG) {
        MeRNSTA.Core.init(window.EVENTS_CONFIG);
    }
    
    console.log('🚀 MeRNSTA Visualizer loaded');
});