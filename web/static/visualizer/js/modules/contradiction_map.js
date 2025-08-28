/**
 * 🧠 MeRNSTA Memory Graph Visualizer - Contradiction Map Module
 * Phase 34 - Interactive contradiction and fact relationship visualization
 */

window.MeRNSTA = window.MeRNSTA || {};

MeRNSTA.ContradictionMap = {
    // Visualization state
    svg: null,
    simulation: null,
    nodes: [],
    links: [],
    nodeElements: null,
    linkElements: null,
    
    // Configuration
    config: {
        width: 800,
        height: 600,
        nodeRadius: 8,
        linkStrength: 0.3,
        chargeStrength: -300,
        centerStrength: 0.1,
        collisionRadius: 20
    },
    
    // Color schemes
    colors: {
        fact: '#00d4ff',
        contradiction: '#ff4444',
        support: '#00ff88',
        neutral: '#888888',
        selected: '#ffaa00'
    },
    
    /**
     * Initialize the contradiction map
     */
    init: function(containerId = 'main-svg') {
        console.log('🔗 Initializing Contradiction Map');
        
        const container = d3.select(`#${containerId}`);
        if (container.empty()) {
            console.error('Container not found:', containerId);
            return;
        }
        
        // Set up SVG
        this.setupSVG(container);
        
        // Set up simulation
        this.setupSimulation();
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Load initial data
        this.loadData();
        
        return this;
    },
    
    /**
     * Set up SVG container
     */
    setupSVG: function(container) {
        // Get container dimensions
        const rect = container.node().getBoundingClientRect();
        this.config.width = rect.width || 800;
        this.config.height = rect.height || 600;
        
        // Clear existing content
        container.selectAll('*').remove();
        
        // Create SVG
        this.svg = container
            .attr('width', this.config.width)
            .attr('height', this.config.height)
            .attr('viewBox', `0 0 ${this.config.width} ${this.config.height}`);
        
        // Add definitions for arrows and patterns
        const defs = this.svg.append('defs');
        
        // Arrow marker for directed edges
        defs.append('marker')
            .attr('id', 'arrowhead')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 15)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', this.colors.neutral);
        
        // Create groups for different elements
        this.svg.append('g').attr('class', 'links');
        this.svg.append('g').attr('class', 'nodes');
        this.svg.append('g').attr('class', 'labels');
        
        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 5])
            .on('zoom', (event) => {
                this.svg.selectAll('g').attr('transform', event.transform);
            });
        
        this.svg.call(zoom);
    },
    
    /**
     * Set up force simulation
     */
    setupSimulation: function() {
        this.simulation = d3.forceSimulation()
            .force('link', d3.forceLink().id(d => d.id).strength(this.config.linkStrength))
            .force('charge', d3.forceManyBody().strength(this.config.chargeStrength))
            .force('center', d3.forceCenter(this.config.width / 2, this.config.height / 2).strength(this.config.centerStrength))
            .force('collision', d3.forceCollide(this.config.collisionRadius))
            .on('tick', () => this.tick());
    },
    
    /**
     * Set up event listeners
     */
    setupEventListeners: function() {
        // Listen for core events
        MeRNSTA.Core.on('autoRefresh', () => this.loadData());
        MeRNSTA.Core.on('resetView', () => this.resetView());
        MeRNSTA.Core.on('exportData', () => this.exportData());
        
        // Listen for control changes
        const controls = ['time-range', 'data-limit', 'physics-enabled'];
        controls.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('change', () => this.loadData());
            }
        });
        
        // Refresh button
        const refreshBtn = document.getElementById('refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.loadData());
        }
    },
    
    /**
     * Load data from API
     */
    async loadData() {
        console.log('📊 Loading contradiction map data');
        MeRNSTA.Core.showLoading('viz-container');
        
        try {
            const timeRange = document.getElementById('time-range')?.value || '24h';
            const limit = parseInt(document.getElementById('data-limit')?.value || '100');
            
            const result = await MeRNSTA.Core.getVisualizerData('contradiction_map', timeRange, limit);
            
            if (result.success) {
                this.updateVisualization(result.data);
                this.updateStats(result.data.metadata);
                MeRNSTA.Core.hideLoading('viz-container');
                MeRNSTA.Core.updateLastUpdate();
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            console.error('Failed to load contradiction map data:', error);
            MeRNSTA.Core.showError('viz-container', error.message);
        }
    },
    
    /**
     * Update visualization with new data
     */
    updateVisualization: function(data) {
        console.log('🎨 Updating contradiction map visualization', data);
        
        // Update data
        this.nodes = data.nodes || [];
        this.links = data.edges || [];
        
        // Update links
        this.updateLinks();
        
        // Update nodes
        this.updateNodes();
        
        // Update simulation
        this.simulation.nodes(this.nodes);
        this.simulation.force('link').links(this.links);
        this.simulation.alpha(1).restart();
    },
    
    /**
     * Update link elements
     */
    updateLinks: function() {
        const linkGroup = this.svg.select('.links');
        
        this.linkElements = linkGroup.selectAll('.link')
            .data(this.links, d => d.id);
        
        // Remove old links
        this.linkElements.exit().remove();
        
        // Add new links
        const linkEnter = this.linkElements.enter()
            .append('line')
            .attr('class', 'link')
            .attr('stroke', d => this.getLinkColor(d))
            .attr('stroke-width', d => this.getLinkWidth(d))
            .attr('opacity', 0.6);
        
        // Merge and update
        this.linkElements = linkEnter.merge(this.linkElements);
        
        // Add hover effects
        this.linkElements
            .on('mouseover', (event, d) => this.showLinkTooltip(event, d))
            .on('mouseout', () => this.hideTooltip());
    },
    
    /**
     * Update node elements
     */
    updateNodes: function() {
        const nodeGroup = this.svg.select('.nodes');
        
        this.nodeElements = nodeGroup.selectAll('.node')
            .data(this.nodes, d => d.id);
        
        // Remove old nodes
        this.nodeElements.exit().remove();
        
        // Add new nodes
        const nodeEnter = this.nodeElements.enter()
            .append('g')
            .attr('class', 'node')
            .call(this.drag());
        
        // Add circles
        nodeEnter.append('circle')
            .attr('r', this.config.nodeRadius)
            .attr('fill', d => this.getNodeColor(d))
            .attr('stroke', '#333')
            .attr('stroke-width', 1);
        
        // Add labels
        nodeEnter.append('text')
            .attr('class', 'node-label')
            .attr('dy', -15)
            .attr('text-anchor', 'middle')
            .attr('font-size', '10px')
            .attr('fill', '#fff')
            .text(d => this.getNodeLabel(d));
        
        // Merge and update
        this.nodeElements = nodeEnter.merge(this.nodeElements);
        
        // Add interaction
        this.nodeElements
            .on('mouseover', (event, d) => this.showNodeTooltip(event, d))
            .on('mouseout', () => this.hideTooltip())
            .on('click', (event, d) => this.selectNode(event, d));
    },
    
    /**
     * Simulation tick function
     */
    tick: function() {
        if (this.linkElements) {
            this.linkElements
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
        }
        
        if (this.nodeElements) {
            this.nodeElements
                .attr('transform', d => `translate(${d.x}, ${d.y})`);
        }
    },
    
    /**
     * Get node color based on type and data
     */
    getNodeColor: function(node) {
        switch (node.type) {
            case 'fact':
                const confidence = node.data.confidence || 0.5;
                return d3.interpolateRgb(this.colors.neutral, this.colors.fact)(confidence);
            case 'contradiction':
                return this.colors.contradiction;
            default:
                return this.colors.neutral;
        }
    },
    
    /**
     * Get link color based on relationship type
     */
    getLinkColor: function(link) {
        switch (link.label) {
            case 'contradicts':
                return this.colors.contradiction;
            case 'supports':
                return this.colors.support;
            default:
                return this.colors.neutral;
        }
    },
    
    /**
     * Get link width based on weight/strength
     */
    getLinkWidth: function(link) {
        const weight = link.weight || 0.5;
        return Math.max(1, weight * 5);
    },
    
    /**
     * Get node label
     */
    getNodeLabel: function(node) {
        return node.label.length > 20 ? node.label.substr(0, 20) + '...' : node.label;
    },
    
    /**
     * Drag behavior
     */
    drag: function() {
        return d3.drag()
            .on('start', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            })
            .on('drag', (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on('end', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            });
    },
    
    /**
     * Show node tooltip
     */
    showNodeTooltip: function(event, node) {
        const tooltip = d3.select('#graph-tooltip');
        const content = `
            <strong>${node.label}</strong><br>
            Type: ${node.type}<br>
            Confidence: ${(node.data.confidence * 100).toFixed(1)}%<br>
            Volatility: ${(node.data.volatility * 100).toFixed(1)}%<br>
            Timestamp: ${new Date(node.data.timestamp).toLocaleString()}
        `;
        
        tooltip.select('#tooltip-content').html(content);
        tooltip.style('display', 'block')
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px');
    },
    
    /**
     * Show link tooltip
     */
    showLinkTooltip: function(event, link) {
        const tooltip = d3.select('#graph-tooltip');
        const content = `
            <strong>${link.label}</strong><br>
            Strength: ${(link.weight * 100).toFixed(1)}%<br>
            Type: ${link.data?.contradiction_type || 'unknown'}<br>
            Discovered: ${new Date(link.data?.discovered_at).toLocaleString()}
        `;
        
        tooltip.select('#tooltip-content').html(content);
        tooltip.style('display', 'block')
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px');
    },
    
    /**
     * Hide tooltip
     */
    hideTooltip: function() {
        d3.select('#graph-tooltip').style('display', 'none');
    },
    
    /**
     * Select node
     */
    selectNode: function(event, node) {
        // Highlight connected nodes and links
        this.nodeElements.classed('selected', false);
        this.linkElements.classed('selected', false);
        
        // Select current node
        d3.select(event.currentTarget).classed('selected', true);
        
        // Highlight connected elements
        const connectedNodes = new Set();
        this.links.forEach(link => {
            if (link.source.id === node.id || link.target.id === node.id) {
                connectedNodes.add(link.source.id);
                connectedNodes.add(link.target.id);
            }
        });
        
        this.nodeElements.classed('connected', d => connectedNodes.has(d.id));
        this.linkElements.classed('connected', d => 
            d.source.id === node.id || d.target.id === node.id);
    },
    
    /**
     * Update statistics display
     */
    updateStats: function(metadata) {
        const statsContainer = document.getElementById('module-stats');
        if (!statsContainer) return;
        
        const stats = [
            { label: 'Total Contradictions', value: metadata.total_contradictions || 0 },
            { label: 'Node Count', value: metadata.node_count || 0 },
            { label: 'Edge Count', value: metadata.edge_count || 0 },
            { label: 'Time Range', value: metadata.time_range || 'unknown' }
        ];
        
        statsContainer.innerHTML = stats.map(stat => `
            <div class="stat-item">
                <span class="stat-label">${stat.label}:</span>
                <span class="stat-value">${stat.value}</span>
            </div>
        `).join('');
    },
    
    /**
     * Reset view
     */
    resetView: function() {
        if (this.svg) {
            this.svg.transition().duration(750).call(
                d3.zoom().transform,
                d3.zoomIdentity
            );
        }
        
        // Clear selections
        this.nodeElements?.classed('selected', false);
        this.linkElements?.classed('selected', false);
        this.nodeElements?.classed('connected', false);
        this.linkElements?.classed('connected', false);
    },
    
    /**
     * Export data
     */
    exportData: function() {
        const data = {
            nodes: this.nodes,
            links: this.links,
            metadata: {
                exported_at: new Date().toISOString(),
                module: 'contradiction_map',
                node_count: this.nodes.length,
                link_count: this.links.length
            }
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `mernsta_contradiction_map_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    if (window.MODULE_CONFIG && window.MODULE_CONFIG.module_name === 'contradiction_map') {
        MeRNSTA.ContradictionMap.init();
    }
});