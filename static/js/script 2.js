document.querySelector('#recommend-btn').addEventListener('click', () => {
    const skills = document.querySelector('#skills-input').value;
    
    fetch('/recommend', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ skills: skills })
    })
    .then(response => response.json())
    .then(data => {
        displayJobs(data.jobs);
    });
});

function displayJobs(jobs) {
    const resultsDiv = document.querySelector('#results');
    resultsDiv.innerHTML = jobs.map(job => `<p>${job}</p>`).join('');
}
