// VRAM Profiler - Web UI for loading and visualizing VRAM usage snapshots.

// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const loading = document.getElementById('loading');
const loadingText = document.getElementById('loading-text');
const tooltip = document.getElementById('tooltip');
const resetBtn = document.getElementById('reset-btn');

const stats = {
    peak: document.getElementById('stat-peak'),
    frag: document.getElementById('stat-frag'),
    duration: document.getElementById('stat-duration'),
    events: document.getElementById('stat-events'),
    bottlenecks: document.getElementById('bottleneck-list'),
    stack: document.getElementById('stack-details')
};

// Global State
let rawData = null;
let normalizedEvents = [];
let timelineData = []; // {t, mem, cached}
let maxMem = 0;
let globalStartTime = 0;
let durationUs = 0;

// D3 Margins globals
const margin = { top: 20, right: 30, bottom: 30, left: 60 };

// --- Interaction Init ---

// Auto-load matches
window.addEventListener('DOMContentLoaded', () => {
    // Check for injected or URL data
    if (window.VRAM_SNAPSHOT_DATA) {
        console.log("Auto-loading from injected snapshot_data.js");
        loading.classList.remove('hidden');
        // Small delay to allow UI to render 'loading' state
        setTimeout(() => {
            rawData = window.VRAM_SNAPSHOT_DATA;
            processAndRender();
        }, 100);
        return;
    }

    // 2. Check for URL param (Server/Hosted mode)
    const params = new URLSearchParams(window.location.search);
    const url = params.get('url');
    if (url) {
        console.log(`Auto-loading from URL: ${url}`);
        loading.classList.remove('hidden');
        loadingText.textContent = `Fetching snapshot from ${url}...`;
        fetch(url)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                rawData = data;
                processAndRender();
            })
            .catch(error => {
                console.error('Error fetching snapshot from URL:', error);
                alert(`Error fetching snapshot from URL: ${error.message}`);
                loading.classList.add('hidden');
            });
        return;
    }
});

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('active'); });
dropZone.addEventListener('dragleave', () => { dropZone.classList.remove('active'); });
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('active');
    handleFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', (e) => { handleFiles(e.target.files); });
resetBtn.addEventListener('click', () => {
    if (normalizedEvents.length > 0) processAndRender();
});

function handleFiles(files) {
    if (files.length === 0) return;
    const file = files[0];

    loading.classList.remove('hidden');
    loadingText.textContent = "Parsing JSON Snapshot...";

    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            rawData = JSON.parse(e.target.result);
            processAndRender();
        } catch (err) {
            console.error(err);
            alert("Error parsing snapshot: " + err.message);
            loading.classList.add('hidden');
        }
    };
    reader.onerror = (e) => {
        console.error("FileReader Error:", e);
        alert("Error reading file. See console for details.");
        loading.classList.add('hidden');
    };
    reader.readAsText(file);
}

// --- Data Normalization & Logic ---

function normalizeEvent(ev) {
    // PyTorch snapshot formats vary between versions.
    if (!ev) return null;

    // Internal Action Codes:
    // 0: Alloc (Active)
    // 1: Free (Active)
    // 2: Segment Alloc (Cached/Reserved)
    // 3: Segment Free (Cached/Reserved)

    const actionMap = {
        'alloc': 0,
        'free_completed': 1,
        'free': 1,
        'segment_alloc': 2,
        'segment_free': 3
    };

    // Format A (Legacy/Simple): [ts, action, addr, size, stream, frame_idx]
    if (Array.isArray(ev)) {
        return {
            t: ev[0],
            action: ev[1],
            addr: ev[2],
            size: ev[3],
            frame: ev[5]
        };
    }

    // Format B (Modern): { time_us, action, addr, size, frames: [...] }
    if (typeof ev === 'object') {
        let act = ev.action;
        if (typeof act === 'string') {
            act = actionMap[act] !== undefined ? actionMap[act] : -1;
        }

        if (act === -1) return null;

        return {
            t: ev.time_us || ev.ts || 0,
            action: act,
            addr: ev.addr,
            size: ev.size || 0,
            frame: ev.frames
        };
    }
    return null;
}

function processAndRender() {
    if (!rawData.device_traces || !rawData.device_traces[0]) {
        alert("Snapshot invalid: missing device_traces.");
        loading.classList.add('hidden');
        return;
    }

    const trace = rawData.device_traces[0];
    normalizedEvents = trace.map(normalizeEvent).filter(e => e !== null);
    normalizedEvents.sort((a, b) => a.t - b.t);

    if (normalizedEvents.length === 0) {
        alert("Trace is empty.");
        loading.classList.add('hidden');
        return;
    }

    globalStartTime = normalizedEvents[0].t;
    durationUs = normalizedEvents[normalizedEvents.length - 1].t - globalStartTime;

    // --- Baseline Reconstruction (Reverse Replay) ---
    // The trace is often truncated (circular buffer), missing initial allocations (weights).
    // We use `segments` (final state) to calculate the Baseline.

    let finalReserved = 0;
    let finalActive = 0;

    if (rawData.segments) {
        rawData.segments.forEach(seg => {
            finalReserved += seg.total_size;
            // Iterate blocks to find active tensors
            if (seg.blocks) {
                seg.blocks.forEach(blk => {
                    if (blk.state === 'active_allocated') {
                        finalActive += blk.size;
                    }
                });
            }
        });
    }

    // Calculate Delta during trace
    let deltaReserved = 0;
    let deltaActive = 0;

    normalizedEvents.forEach(ev => {
        if (ev.action === 0) deltaActive += ev.size;       // Alloc
        else if (ev.action === 1) deltaActive -= ev.size;  // Free
        else if (ev.action === 2) deltaReserved += ev.size;// Seg Alloc
        else if (ev.action === 3) deltaReserved -= ev.size;// Seg Free
    });

    // Baseline = Final - Delta
    // Use Math.max(0, ...) to avoid negative baseline if trace > snapshot (edge case)
    const startReserved = Math.max(0, finalReserved - deltaReserved);
    const startActive = Math.max(0, finalActive - deltaActive);

    console.log(`Baseline Reconstruction:
    Final Reserved: ${(finalReserved / 1024 ** 3).toFixed(2)} GB
    Final Active:   ${(finalActive / 1024 ** 3).toFixed(2)} GB
    Trace Delta R:  ${(deltaReserved / 1024 ** 3).toFixed(2)} GB
    Trace Delta A:  ${(deltaActive / 1024 ** 3).toFixed(2)} GB
    Start Baseline R: ${(startReserved / 1024 ** 3).toFixed(2)} GB
    Start Baseline A: ${(startActive / 1024 ** 3).toFixed(2)} GB`);

    // --- Forward Pass with Baseline ---
    let currentActive = startActive;
    let currentReserved = startReserved;

    maxMem = 0;
    timelineData = [];

    const activeAllocMap = new Map();
    const cachedSegmentMap = new Map();

    // Add baseline point
    timelineData.push({ t: 0, active: currentActive, total: currentReserved });

    normalizedEvents.forEach(ev => {
        const relTime = ev.t - globalStartTime;

        switch (ev.action) {
            case 0: // ALLOC
                activeAllocMap.set(ev.addr, ev.size);
                currentActive += ev.size;
                break;
            case 1: // FREE
                const sz = activeAllocMap.get(ev.addr) || 0;
                currentActive -= sz;
                activeAllocMap.delete(ev.addr);
                break;
            case 2: // SEGMENT ALLOC
                cachedSegmentMap.set(ev.addr, ev.size);
                currentReserved += ev.size;
                break;
            case 3: // SEGMENT FREE
                const segSz = cachedSegmentMap.get(ev.addr) || 0;
                currentReserved -= segSz;
                cachedSegmentMap.delete(ev.addr);
                break;
        }

        // Use Reserved as the total VRAM footprint
        const effectiveMem = currentReserved;

        if (effectiveMem > maxMem) maxMem = effectiveMem;

        timelineData.push({
            t: relTime,
            active: currentActive,
            total: effectiveMem
        });
    });

    // Update Dashboard Stats
    stats.peak.textContent = (maxMem / (1024 ** 3)).toFixed(2) + " GB";
    stats.duration.textContent = (durationUs / 1e6).toFixed(2) + " s";
    stats.events.textContent = normalizedEvents.length.toLocaleString();

    // Calculate Fragmentation at Peak
    // Identify point of maxMem
    const peakPoint = timelineData.find(d => d.total === maxMem);
    if (peakPoint && peakPoint.total > 0) {
        const frag = (peakPoint.total - peakPoint.active) / peakPoint.total * 100;
        stats.frag.textContent = frag.toFixed(1) + "%";
        // Color code
        if (frag > 20) stats.frag.style.color = "#f56565"; // Red
        else if (frag > 10) stats.frag.style.color = "#ecc94b"; // Yellow
        else stats.frag.style.color = "#48bb78"; // Green
    }

    // Find Bottlenecks (Top Active Allocations)
    findBottlenecks(activeAllocMap, normalizedEvents);

    // Initial Renders
    renderTimeline();

    // Default to showing state at Peak
    // Identify the index of peak
    const peakIdx = timelineData.findIndex(d => d.total === maxMem);
    const peakTime = timelineData[peakIdx].t;
    reconstructAndRenderFlame(peakTime);

    // Display Metadata if available
    if (rawData.user_metadata) {
        const m = rawData.user_metadata;
        loadingText.textContent = `Loaded snapshot for ${m.gpu} (${m.context} context)`;

        document.title = `VRAM Profile - ${m.gpu}`;
    }

    loading.classList.add('hidden');
}

// Search Binding
const searchInput = document.getElementById('sidebar-search');
searchInput.addEventListener('input', (e) => {
    // Re-run finding bottlenecks with filter
    findBottlenecks(activeAllocMapRef, normalizedEvents, e.target.value);
});
// Need to keep a ref to the map for re-filtering
let activeAllocMapRef = null;

function findBottlenecks(currentActive, trace, filter = "") {
    activeAllocMapRef = currentActive; // Store for search

    const stackAggregator = new Map();

    trace.forEach(ev => {
        // Look for large Allocations (Active)
        if (ev.action === 0 && ev.size > 10 * 1024 * 1024) {
            const key = JSON.stringify(ev.frame);
            const entry = stackAggregator.get(key) || { bytes: 0, frame: ev.frame, count: 0 };
            entry.bytes = Math.max(entry.bytes, ev.size);
            entry.count++;
            stackAggregator.set(key, entry);
        }
    });

    let sorted = Array.from(stackAggregator.values())
        .sort((a, b) => b.bytes - a.bytes);


    // Apply Filter
    if (filter) {
        const lower = filter.toLowerCase();
        sorted = sorted.filter(item => {
            const name = resolveStack(item.frame, 5).toLowerCase();
            return name.includes(lower);
        });
    }

    // Slice after filter
    const displayList = sorted.slice(0, 20);

    stats.bottlenecks.innerHTML = "";
    if (displayList.length === 0) {
        stats.bottlenecks.innerHTML = "<p style='color:var(--text-dim)'>No matches found.</p>";
        return;
    }

    displayList.forEach(item => {
        const displayFrame = resolveStack(item.frame, 1);
        const div = document.createElement('div');
        div.className = "bottleneck-item";
        div.innerHTML = `
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span class="bottleneck-size">${(item.bytes / 1024 ** 2).toFixed(1)} MB</span>
                <span style="color:var(--text-dim); font-size: 0.7rem;">Seen ${item.count}x</span>
            </div>
            <div style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${displayFrame}</div>
        `;
        div.onclick = () => {
            searchInput.value = ""; // Optional: clear search on click? kept for now.
            stats.stack.innerHTML = `<div><strong style="color:var(--accent)">Max Single Alloc: ${(item.bytes / 1024 ** 2).toFixed(2)} MB</strong></div>` +
                resolveStack(item.frame, 25).replace(/ < /g, '\n<span style="color:var(--text-dim)">▲</span> ');
        };
        stats.bottlenecks.appendChild(div);
    });
}

// --- Rendering Engines ---

const toggleReserved = document.getElementById('toggle-reserved');
toggleReserved.addEventListener('change', () => {
    if (timelineData.length > 0) renderTimeline();
});

function renderTimeline() {
    const showReserved = toggleReserved.checked;
    const parent = document.getElementById('timeline-host');
    const container = document.getElementById('timeline');
    container.innerHTML = '';

    const w = parent.offsetWidth;
    // Fix: Use bounding client rect or fallback to ensure non-zero height
    let h = parent.offsetHeight;
    if (h < 100) h = 300; // Force min height if not detected

    const chartW = w - margin.left - margin.right;
    const chartH = h - margin.top - margin.bottom;

    if (chartH <= 0) {
        console.error("Chart height is too small to render:", chartH);
        return;
    }

    const svg = d3.select("#timeline")
        .append("svg")
        .attr("width", w)
        .attr("height", h);

    // Clip Path (to prevent drawing outside axes during zoom)
    svg.append("defs").append("clipPath")
        .attr("id", "clip")
        .append("rect")
        .attr("width", chartW)
        .attr("height", chartH);

    const g = svg.append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // Scales
    const x = d3.scaleLinear().domain([0, durationUs]).range([0, chartW]);
    const y = d3.scaleLinear().domain([0, maxMem * 1.1]).range([chartH, 0]);

    // Downsample for performance (if too large)
    let drawData = timelineData;
    if (timelineData.length > 5000) {
        const factor = Math.ceil(timelineData.length / 5000);
        drawData = timelineData.filter((_, i) => i % factor === 0);
    }

    // Axes
    const xAxisGroups = g.append("g")
        .attr("transform", `translate(0,${chartH})`)
        .call(d3.axisBottom(x).ticks(8).tickFormat(d => (d / 1e6).toFixed(1) + "s"));

    const yAxisGroup = g.append("g")
        .call(d3.axisLeft(y).ticks(5).tickFormat(d => (d / 1e9).toFixed(1) + " GB"));

    // Styling Axes
    xAxisGroups.selectAll("text").attr("fill", "#718096");
    xAxisGroups.selectAll("line").attr("stroke", "#4a5568");
    xAxisGroups.select(".domain").attr("stroke", "#4a5568");

    yAxisGroup.selectAll("text").attr("fill", "#718096");
    yAxisGroup.selectAll("line").attr("stroke", "#4a5568");
    yAxisGroup.select(".domain").attr("stroke", "#4a5568");

    // Generators
    const areaTotal = d3.area()
        .x(d => x(d.t))
        .y0(chartH)
        .y1(d => y(d.total))
        .curve(d3.curveStepAfter);

    const areaActive = d3.area()
        .x(d => x(d.t))
        .y0(chartH)
        .y1(d => y(d.active))
        .curve(d3.curveStepAfter);

    // Paths Container (Clipped)
    const content = g.append("g").attr("clip-path", "url(#clip)");

    let pathTotal;
    if (showReserved) {
        pathTotal = content.append("path")
            .datum(drawData)
            .attr("fill", "#64748b") // Lighter cool gray for better contrast
            .attr("opacity", 0.7) // Increased opacity
            .attr("d", areaTotal);
    }

    const pathActive = content.append("path")
        .datum(drawData)
        .attr("fill", "url(#gradient)")
        .attr("stroke", varProp("--accent"))
        .attr("stroke-width", 1.5)
        .attr("d", areaActive);

    // Gradient Definition
    const defs = svg.append("defs");
    const grad = defs.append("linearGradient").attr("id", "gradient").attr("x1", "0%").attr("y1", "0%").attr("x2", "0%").attr("y2", "100%");
    grad.append("stop").attr("offset", "0%").attr("stop-color", varProp("--accent")).attr("stop-opacity", 0.5);
    grad.append("stop").attr("offset", "100%").attr("stop-color", varProp("--accent")).attr("stop-opacity", 0);

    // Interaction Components
    const focus = g.append("line")
        .attr("y1", 0).attr("y2", chartH)
        .attr("stroke", "#fff")
        .attr("stroke-width", 1)
        .style("opacity", 0)
        .style("pointer-events", "none"); // Let events pass to rect

    // Zoom Behavior
    const zoom = d3.zoom()
        .scaleExtent([1, 500]) // 1x to 500x zoom
        .translateExtent([[0, 0], [chartW, chartH]])
        .extent([[0, 0], [chartW, chartH]])
        .on("zoom", updateChart);

    // Overlay Rect for Events (Zoom + Click)
    const overlay = svg.append("rect")
        .attr("width", w) // Cover entire SVG to catch zoom events anywhere
        .attr("height", h)
        .style("fill", "none")
        .style("pointer-events", "all")
        .call(zoom)
        .on("mousemove", handleMouseMove)
        .on("mouseout", () => {
            focus.style("opacity", 0);
            tooltip.style.display = 'none';
        })
        .on("click", handleClick);

    let newX = x; // Current scale reference

    function updateChart(event) {
        newX = event.transform.rescaleX(x);

        // Update Axes
        xAxisGroups.call(d3.axisBottom(newX).ticks(8).tickFormat(d => (d / 1e6).toFixed(1) + "s"));
        xAxisGroups.selectAll("text").attr("fill", "#718096");

        // Update Paths
        if (pathTotal) pathTotal.attr("d", areaTotal.x(d => newX(d.t)));
        pathActive.attr("d", areaActive.x(d => newX(d.t)));
    }

    function handleMouseMove(event) {
        // We need mouse pos relative to 'g' (the chart area)
        // But the event target is the overlay rect which covers the whole SVG.
        // d3.pointer returns [x, y] relative to target.
        // We need to adjust by margin.left if we are using the overlay rect on top of SVG.

        const [mx, my] = d3.pointer(event);
        const chartX = mx - margin.left;

        // Check bounds
        if (chartX < 0 || chartX > chartW) return;

        const t = newX.invert(chartX); // Use transformed scale

        focus.attr("x1", chartX).attr("x2", chartX).style("opacity", 0.8);

        tooltip.style.display = 'block';
        tooltip.style.left = (event.pageX + 15) + 'px';
        tooltip.style.top = (event.pageY - 15) + 'px';

        const idx = d3.bisectLeft(timelineData.map(d => d.t), t);
        const point = timelineData[Math.min(idx, timelineData.length - 1)] || timelineData[timelineData.length - 1];

        if (point) {
            let reservedHtml = "";
            if (showReserved) {
                reservedHtml = `Reserved: <b>${(point.total / 1024 ** 3).toFixed(2)} GB</b><br>`;
            }

            tooltip.innerHTML = `
                Time: <b>${(t / 1e6).toFixed(3)}s</b><br>
                ${reservedHtml}
                Active: <b style="color:var(--accent)">${(point.active / 1024 ** 3).toFixed(2)} GB</b>
            `;
        }
    }

    function handleClick(event) {
        const [mx] = d3.pointer(event);
        const chartX = mx - margin.left;
        if (chartX < 0 || chartX > chartW) return;

        const t = newX.invert(chartX);
        reconstructAndRenderFlame(t);
    }
}




function reconstructAndRenderFlame(time) {
    const active = new Map();
    normalizedEvents.forEach(ev => {
        if ((ev.t - globalStartTime) > time) return;
        if (ev.action === 0) active.set(ev.addr, ev);
        else if (ev.action === 1) active.delete(ev.addr);
    });

    // Group by stack
    const groups = new Map();
    active.forEach(ev => {
        const key = JSON.stringify(ev.frame);
        const g = groups.get(key) || { bytes: 0, frame: ev.frame, count: 0 };
        g.bytes += ev.size;
        g.count++;
        groups.set(key, g);
    });

    const data = Array.from(groups.values()).sort((a, b) => b.bytes - a.bytes);

    const container = document.getElementById('flamegraph');
    container.innerHTML = "";

    if (data.length === 0) {
        container.innerHTML = `<p style="color: var(--text-dim); text-align: center; padding: 40px;">No memory allocated at this timestamp (${(time / 1e6).toFixed(3)}s).</p>`;
        return;
    }

    const maxActive = data[0].bytes;

    data.forEach(item => {
        const row = document.createElement("div");
        row.style.height = "24px";
        row.style.position = "relative";
        row.style.background = "#000";
        row.style.cursor = "pointer";
        row.style.marginBottom = "2px";
        row.style.borderRadius = "4px";
        row.className = "flame-bar-row";
        row.tabIndex = 0;

        const bar = document.createElement("div");
        bar.style.height = "100%";
        bar.style.width = (item.bytes / maxActive * 100) + "%";
        bar.style.background = hashColor(resolveStack(item.frame, 1));
        bar.style.opacity = 0.8;
        bar.style.borderRadius = "4px";

        const label = document.createElement("span");
        label.style.position = "absolute";
        label.style.left = "8px";
        label.style.top = "4px";
        label.style.fontSize = "11px";
        label.style.color = "#000";
        label.style.fontWeight = "600";
        label.style.textShadow = "0 0 4px rgba(255,255,255,0.8)";
        label.style.whiteSpace = "nowrap";
        label.style.overflow = "hidden";
        label.textContent = `${(item.bytes / 1024 ** 2).toFixed(1)} MB | ${resolveStack(item.frame, 1)}`;

        row.appendChild(bar);
        row.appendChild(label);

        const selectRow = () => {
            document.querySelectorAll('.flame-bar-row').forEach(r => r.classList.remove('active-row'));
            row.classList.add('active-row');
            stats.stack.innerHTML = `<div style="color:var(--accent); margin-bottom:10px; font-weight:700;">Stack Depth: ${item.count > 1 ? item.count + " locations" : "1 location"}</div>` +
                resolveStack(item.frame, 25).replace(/ < /g, '\n<span style="color:var(--text-dim)">▲</span> ');
        };

        row.onclick = selectRow;

        row.onkeydown = (e) => {
            if (e.key === 'Enter' || e.key === ' ' || e.key === 'Spacebar') {
                e.preventDefault();
                selectRow();
            }
        };

        container.appendChild(row);
    });
}

// --- Helpers ---

function resolveStack(frameData, depth = 3) {
    if (frameData === undefined || frameData === null) return "Unknown Trace";

    // Format A: Global Index
    if (typeof frameData === 'number' && rawData.stack_frames) {
        let f = rawData.stack_frames[frameData];
        return f ? `${f[2]} (${f[0].split('/').pop()}:${f[1]})` : "Unknown Frame";
    }

    // Format B: Embedded List
    if (Array.isArray(frameData)) {
        return frameData.slice(0, depth).map(f => {
            const name = f.name || "???";
            const file = (f.filename || f.file || "").split('/').pop();
            return `${name} [${file}:${f.line}]`;
        }).join(' < ');
    }

    return "Native Memory (C++)";
}

function hashColor(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) hash = str.charCodeAt(i) + ((hash << 5) - hash);
    const hue = Math.abs(hash % 360);
    return `hsl(${hue}, 65%, 75%)`;
}

function varProp(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
