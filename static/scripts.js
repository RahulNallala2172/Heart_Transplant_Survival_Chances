document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultsDiv = document.getElementById('results');

    // Add animated gradient background to body
    document.body.classList.add('animated-bg');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => {
            data[key] = value;
        });

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const resp = await response.json();

            if (response.ok) {
                // User-friendly labels
                const labels = {
                    'Chance of surviving 1 year (%)': 'Chance of surviving 1 year',
                    'Chance of surviving 5 years (%)': 'Chance of surviving 5 years',
                    'Chance of surviving 10 years (%)': 'Chance of surviving 10 years',
                    'Risk of rejection within 1 year (%)': 'Risk of rejection within 1 year',
                    'Predicted quality of life score': 'Predicted quality of life score',
                    'Predicted rehospitalizations in 1 year': 'Predicted rehospitalizations in 1 year'
                };

                // Sort keys in desired order
                const orderedKeys = [
                    'Chance of surviving 1 year (%)',
                    'Chance of surviving 5 years (%)',
                    'Chance of surviving 10 years (%)',
                    'Predicted quality of life score',
                    'Predicted rehospitalizations in 1 year',
                    'Risk of rejection within 1 year (%)'
                ];

                let html = '<h3 class="results-title">Predictions</h3><ul class="results-list">';
                for (const key of orderedKeys) {
                    if (key in resp) {
                        let label = labels[key] || key;
                        let val = (typeof resp[key] === 'number') ? Math.round(resp[key] * 100) / 100 : resp[key];
                        if (key.includes('(%)')) {
                            val = val + '%';
                        }
                        html += `<li class="result-item"><span class="result-label">${label}:</span> <span class="result-value">${val}</span></li>`;
                    }
                }
                html += '</ul>';

                // ------------------------------
                // Add suggestion message
                // ------------------------------
                const rejectionRisk = resp["Risk of rejection within 1 year (%)"];
                let suggestion = "";

                if (rejectionRisk > 70) {
                    suggestion = "⚠️ High rejection risk! Close monitoring and preventive care are strongly recommended.";
                } else if (rejectionRisk > 40) {
                    suggestion = "⚠️ Moderate rejection risk. Regular follow-up and lifestyle adjustments advised.";
                } else {
                    suggestion = "✅ Low rejection risk. Continue routine monitoring and healthy lifestyle.";
                }

                html += `<div class="suggestion"><h3>Suggestion</h3><p>${suggestion}</p></div>`;

                // Render results
                resultsDiv.innerHTML = html;
                resultsDiv.style.display = 'block';
                resultsDiv.classList.add('fade-in-card');
                resultsDiv.scrollIntoView({ behavior: 'smooth' });

            } else {
                alert("Error: " + (resp.error || "Unknown error"));
            }

        } catch (err) {
            alert("Prediction request failed: " + err.message);
        }
    });
});
