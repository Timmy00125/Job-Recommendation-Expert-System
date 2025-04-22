document.addEventListener("DOMContentLoaded", function () {
  const jobForm = document.getElementById("job-form");
  const loadingSpinner = document.getElementById("loading-spinner");
  const resultsContainer = document.getElementById("results-container");

  if (jobForm) {
    jobForm.addEventListener("submit", function (e) {
      e.preventDefault();

      // Show loading spinner
      loadingSpinner.classList.remove("hidden");
      resultsContainer.classList.add("hidden");

      // Collect form data
      const formData = {
        location: document.getElementById("location").value,
        it_preference: document.getElementById("it_preference").value,
        user_text: document.getElementById("user_text").value,
      };

      // Send AJAX request to get recommendations
      fetch("/api/recommend", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      })
        .then((response) => response.json())
        .then((data) => {
          // Hide loading spinner
          loadingSpinner.classList.add("hidden");

          // Build results HTML
          let resultsHTML = `
                    <div class="card">
                        <div class="results-header">
                            <h2>Results</h2>
                            <button class="btn btn-secondary" id="new-search-btn">
                                <i class="fas fa-arrow-left"></i> New Search
                            </button>
                        </div>
                        
                        <div class="table-responsive">
                            <table class="results-table">
                                <thead>
                                    <tr>
                                        <th>Title</th>
                                        <th>Company</th>
                                        <th>Location</th>
                                        <th>Match Score</th>
                                    </tr>
                                </thead>
                                <tbody>
                `;

          // Add each result row
          data.forEach((job) => {
            const scorePercent = Math.round(job.score * 100);
            resultsHTML += `
                        <tr>
                            <td>${job.Title}</td>
                            <td>${job.Company}</td>
                            <td><i class="fas fa-map-marker-alt"></i> ${
                              job.Location
                            }</td>
                            <td>
                                <div class="score-bar">
                                    <div class="score-fill" style="width: ${scorePercent}%"></div>
                                </div>
                                <span class="score-text">${job.score.toFixed(
                                  2
                                )}</span>
                            </td>
                        </tr>
                    `;
          });

          resultsHTML += `
                            </tbody>
                        </table>
                    </div>
                </div>`;

          // Display results
          resultsContainer.innerHTML = resultsHTML;
          resultsContainer.classList.remove("hidden");

          // Set up new search button
          document
            .getElementById("new-search-btn")
            .addEventListener("click", function () {
              resultsContainer.classList.add("hidden");
              jobForm.reset();
            });
        })
        .catch((error) => {
          console.error("Error:", error);
          loadingSpinner.classList.add("hidden");
          alert(
            "An error occurred while fetching recommendations. Please try again."
          );
        });
    });
  }
});
