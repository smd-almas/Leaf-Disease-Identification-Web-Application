let model = null;
// Update these class names with your actual classes
const CLASS_NAMES = ['Normal', 'Virus', 'Nematodes', 'Fungi', 'Bacteria'];

// Load model from model folder
async function loadModel() {
    model = await tf.loadLayersModel('model/model.json');
}

// Image input handler
document.getElementById('leafInput').addEventListener('change', function(e) {
    const reader = new FileReader();
    reader.onload = function() {
        const img = document.getElementById('imagePreview');
        img.src = reader.result;
        img.style.display = 'block';
    }
    reader.readAsDataURL(e.target.files[0]);
});

// Prediction function
async function predict() {
    if (!model) {
        alert('Model is still loading, please wait...');
        return;
    }

    const image = document.getElementById('imagePreview');
    if (!image.src || image.src === window.location.href) {
        alert('Please select an image first!');
        return;
    }

    try {
        // Preprocess image (adjust dimensions if needed)
        const tensor = tf.browser.fromPixels(image)
            .resizeNearestNeighbor([224, 224]) // Typical size for TM models
            .toFloat()
            .div(255.0)
            .expandDims();

        // Make prediction
        const predictions = await model.predict(tensor).data();
        const resultDiv = document.getElementById('result');
        
        // Process results
        const maxProbability = Math.max(...predictions);
        const predictedClassIndex = predictions.indexOf(maxProbability);
        const predictedClassName = CLASS_NAMES[predictedClassIndex];

        // Display results
        resultDiv.innerHTML = `
            <strong>Diagnosis:</strong> ${predictedClassName}<br>
            <strong>Confidence:</strong> ${(maxProbability * 100).toFixed(2)}%
        `;
        
        // Update styling class
        resultDiv.className = predictedClassName === 'Healthy' ? 
            'result healthy' : 'result disease';
        resultDiv.style.display = 'block';

    } catch (error) {
        console.error('Prediction error:', error);
        alert('Error processing image. Please try again.');
    }
}

// Initialize
document.getElementById('predictBtn').addEventListener('click', predict);
loadModel().catch(error => {
    console.error('Model loading error:', error);
    alert('Failed to load model! Check console for details.');
});