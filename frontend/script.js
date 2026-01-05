async function predict() {
    const fileInput = document.getElementById("fileInput");

    if (!fileInput || fileInput.files.length === 0) {
        alert("Please select an image first");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    document.getElementById("result").innerText = "Predicting...";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Server error");
        }

        const data = await response.json();

        document.getElementById("result").innerText =
            `Prediction: ${data.class} (Confidence: ${(data.confidence * 100).toFixed(2)}%)`;
    } catch (error) {
        document.getElementById("result").innerText =
            "❌ Error while predicting";
        console.error(error);
    }
}
