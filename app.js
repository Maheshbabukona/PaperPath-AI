// Wait for the HTML document to be fully loaded before running the script
document.addEventListener('DOMContentLoaded', () => {

    // --- 1. Get DOM Element References ---
    // We get these once and store them in constants for performance
    const papersContainer = document.getElementById('papersContainer');
    const searchBox = document.getElementById('searchBox');
    const topicFilter = document.getElementById('topicFilter');
    const difficultyFilter = document.getElementById('difficultyFilter');
    const priorityFilter = document.getElementById('priorityFilter');
    const resultsCountEl = document.getElementById('resultsCount');
    const completedCountEl = document.getElementById('completedCount');

    // --- 2. Initialize State ---
    let filteredPapers = [...papers]; // Start with all papers
    let completedPapers = new Set(
        JSON.parse(localStorage.getItem('completedPapers') || '[]')
    );

    // --- 3. Define Core Functions ---

    /**
     * Renders the list of filtered papers to the DOM.
     * Groups papers by phase and creates grids.
     */
    function renderPapers() {
        // Clear the container
        papersContainer.innerHTML = '';

        if (filteredPapers.length === 0) {
            papersContainer.innerHTML = '<div class="no-results">No papers found matching your filters.</div>';
            updateStats();
            return;
        }

        // Use a DocumentFragment for performance.
        // We build the entire list in memory *before*
        // adding it to the DOM, which is much faster.
        const fragment = new DocumentFragment();
        let currentPhase = '';
        let currentGrid = null;

        filteredPapers.forEach(paper => {
            // If we're in a new phase, create a header and a new grid
            if (paper.phase !== currentPhase) {
                currentPhase = paper.phase;
                
                // Add phase header
                const phaseHeader = document.createElement('div');
                phaseHeader.className = 'phase-header';
                phaseHeader.innerHTML = `
                    <h2 class="phase-title">${paper.phase}</h2>
                    <p class="phase-description">${paper.phaseDescription}</p>
                `;
                fragment.appendChild(phaseHeader);

                // Create new grid for this phase
                currentGrid = document.createElement('div');
                currentGrid.className = 'paper-grid';
                fragment.appendChild(currentGrid);
            }

            // Create the paper card
            const card = document.createElement('a');
            card.href = paper.link;
            card.target = '_blank';
            card.rel = 'noopener noreferrer';
            card.className = 'paper-card';
            if (completedPapers.has(paper.number)) {
                card.classList.add('completed');
            }

            card.innerHTML = `
                <span class="paper-number">#${paper.number}</span>
                <h3 class="paper-title">${paper.title}</h3>
                <p class="paper-description">${paper.description}</p>
                <div class="paper-tags">
                    ${paper.mandatory ? '<span class="tag mandatory">MANDATORY</span>' : ''}
                    ${paper.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                </div>
                <span class="difficulty ${paper.difficulty}">${paper.difficulty.toUpperCase()}</span>
                <span class="open-icon">↗</span>
            `;

            // Add click listener to mark as complete
            card.addEventListener('click', (e) => {
                // Allow opening in new tab (Ctrl/Cmd + Click) without marking complete
                if (e.ctrlKey || e.metaKey) return;
                
                // Use setTimeout to allow the link to open before re-rendering
                setTimeout(() => {
                    completedPapers.add(paper.number);
                    card.classList.add('completed');
                    updateStats();
                    localStorage.setItem('completedPapers', JSON.stringify([...completedPapers]));
                }, 100);
            });

            // Add the card to the current phase's grid
            currentGrid.appendChild(card);
        });

        // Finally, append the entire fragment to the DOM
        papersContainer.appendChild(fragment);
        updateStats();
    }

    /**
     * Updates the stat counters for results and completed papers.
     */
    function updateStats() {
        resultsCountEl.textContent = `${filteredPapers.length} papers`;
        completedCountEl.textContent = `${completedPapers.size} completed`;
    }

    /**
     * Applies all active filters to the main 'papers' list
     * and triggers a re-render.
     */
    function applyFilters() {
        const searchTerm = searchBox.value.toLowerCase();
        const topic = topicFilter.value;
        const difficulty = difficultyFilter.value;
        const priority = priorityFilter.value;

        filteredPapers = papers.filter(paper => {
            // 1. Search Filter
            const matchesSearch = searchTerm === '' ||
                                  paper.title.toLowerCase().includes(searchTerm) ||
                                  paper.description.toLowerCase().includes(searchTerm) ||
                                  paper.tags.some(tag => tag.toLowerCase().includes(searchTerm));

            // 2. Topic Filter
            const matchesTopic = topic === 'all' || paper.tags.includes(topic);

            // 3. Difficulty Filter
            const matchesDifficulty = difficulty === 'all' || paper.difficulty === difficulty;

            // 4. Priority Filter
            const matchesPriority = priority === 'all' ||
                                    (priority === 'mandatory' && paper.mandatory) ||
                                    (priority === 'optional' && !paper.mandatory);

            return matchesSearch && matchesTopic && matchesDifficulty && matchesPriority;
        });

        // Re-render the page with the filtered list
        renderPapers();
    }

    // --- 4. Attach Event Listeners ---
    searchBox.addEventListener('input', applyFilters);
    topicFilter.addEventListener('change', applyFilters);
    difficultyFilter.addEventListener('change', applyFilters);
    priorityFilter.addEventListener('change', applyFilters);

    // --- 5. Initial Page Load ---
    renderPapers(); // Render all papers on initial load
});