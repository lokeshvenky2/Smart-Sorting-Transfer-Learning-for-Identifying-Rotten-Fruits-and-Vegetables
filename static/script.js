document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const previewImage = document.getElementById('previewImage');
    const predictButton = document.getElementById('predictButton');
    const predictedClassSpan = document.getElementById('predictedClass');
    const confidenceSpan = document.getElementById('confidence');
    const resultsDiv = document.querySelector('.prediction-results');
    const loadingSpinner = document.getElementById('loading');
    const errorMessage = document.getElementById('errorMessage');

    imageUpload.addEventListener('change', () => {
        const file = imageUpload.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = e => {
                previewImage.src = e.target.result;
                previewImage.style.display = 'block';
                predictButton.disabled = false;
            };
            reader.readAsDataURL(file);
        }
    });

    predictButton.addEventListener('click', () => {
        const file = imageUpload.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file); // Flask expects 'file' key

        loadingSpinner.style.display = 'block';
        errorMessage.style.display = 'none';
        resultsDiv.style.display = 'none';

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(res => {
            loadingSpinner.style.display = 'none';
            if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
            return res.json();
        })
        .then(data => {
            if (data.error) {
                errorMessage.textContent = `❌ ${data.error}`;
                errorMessage.style.display = 'block';
                return;
            }
            predictedClassSpan.textContent = data.predicted_class || 'Unknown';
            confidenceSpan.textContent = `${parseFloat(data.confidence).toFixed(2)}%`;

            resultsDiv.style.display = 'block';
        })
        .catch(err => {
            loadingSpinner.style.display = 'none';
            errorMessage.textContent = "❌ Failed to classify image. " + err.message;
            errorMessage.style.display = 'block';
        });
    });
});
