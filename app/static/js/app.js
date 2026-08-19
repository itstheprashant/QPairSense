const q1 = document.getElementById("question1");
const q2 = document.getElementById("question2");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");
const result = document.getElementById("result");
const errorBox = document.getElementById("errorBox");

function updateCount(textarea, counter) {
    document.getElementById(counter).textContent = textarea.value.length;
}

q1.addEventListener("input", () => updateCount(q1, "count1"));
q2.addEventListener("input", () => updateCount(q2, "count2"));

clearBtn.addEventListener("click", () => {
    q1.value = "";
    q2.value = "";
    result.classList.add("hidden");
    errorBox.classList.add("hidden");
    updateCount(q1, "count1");
    updateCount(q2, "count2");
});

analyzeBtn.addEventListener("click", async () => {
    errorBox.classList.add("hidden");

    const question1 = q1.value.trim();
    const question2 = q2.value.trim();

    if (!question1 || !question2) {
        errorBox.textContent = "Please enter both questions.";
        errorBox.classList.remove("hidden");
        return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Analyzing...";

    try {
        const response = await fetch("/api/v1/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({question1, question2})
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Prediction failed.");
        }

        const confidence = data.confidence * 100;

        document.getElementById("resultBadge").textContent =
            data.is_duplicate ? "DUPLICATE" : "NOT DUPLICATE";
        document.getElementById("resultTitle").textContent = data.label;
        document.getElementById("resultMessage").textContent = data.message;
        document.getElementById("confidenceValue").textContent =
            confidence.toFixed(2) + "%";
        document.getElementById("confidenceBar").style.width =
            confidence.toFixed(2) + "%";
        document.getElementById("duplicateProbability").textContent =
            (data.duplicate_probability * 100).toFixed(2) + "%";
        document.getElementById("nonDuplicateProbability").textContent =
            (data.non_duplicate_probability * 100).toFixed(2) + "%";

        result.classList.remove("hidden");
    } catch (error) {
        errorBox.textContent = error.message;
        errorBox.classList.remove("hidden");
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = "Analyze Questions";
    }
});
