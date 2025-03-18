// Sentences Data
const sentences = [
    "Hello, my name is [Name].",
    "I am [Name], present today.",
    "Yes, I am here.",
    "This is [Name] speaking.",
    "[Name] here, marking my attendance.",
    "Present and ready.",
    "Good morning, my name is [Name].",
    "How are you doing today?",
    "It’s a great day outside.",
    "Can you hear me clearly?",
    "I love learning new things.",
    "Let’s get started with today’s class.",
    "This is just a test sentence.",
    "Please repeat that once again.",
    "I hope everyone is doing well.",
    "The quick brown fox jumps over the lazy dog.",
    "I enjoy listening to music in my free time.",
    "Artificial intelligence is changing the world.",
    "A rolling stone gathers no moss.",
    "Technology is advancing at a rapid pace.",
    "Speech recognition is an interesting field of study.",
    "She sells seashells by the seashore.",
    "A journey of a thousand miles begins with a single step.",
    "My phone number is 9876543210.",
    "The time now is 10:30 AM.",
    "Today’s date is the 5th of March.",
    "I will be 22 years old next year.",
    "Yes.",
    "No.",
    "Okay.",
    "Hmm.",
    "Alright.",
    "Thank you."
];

// Recording State
let recordings = Array(sentences.length).fill(null);
let activeRecorder = null;

// Create Sentence Cards
sentences.forEach((text, index) => {
    const card = document.createElement("div");
    card.className = "sentence-card";
    card.innerHTML = `
        <p class="sentence-text">${text}</p>
        <div class="controls">
            <button class="startBtn" data-index="${index}">🎤 Start</button>
            <button class="stopBtn" data-index="${index}" disabled>⏹️ Stop</button>
        </div>
        <audio class="preview" controls hidden></audio>
    `;
    document.getElementById("sentencesGrid").appendChild(card);

    // Elements
    const startBtn = card.querySelector(".startBtn");
    const stopBtn = card.querySelector(".stopBtn");
    const audioPreview = card.querySelector(".preview");

    // Recording Handlers
    startBtn.addEventListener("click", async () => {
        try {
            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const recorder = new MediaRecorder(stream);
            const chunks = [];

            recorder.ondataavailable = (e) => chunks.push(e.data);
            recorder.onstop = () => {
                const blob = new Blob(chunks, { type: "audio/wav" });
                recordings[index] = blob;
                audioPreview.src = URL.createObjectURL(blob);
                audioPreview.hidden = false;
                checkCompletion();
            };

            recorder.start();
            activeRecorder = recorder;
            startBtn.disabled = true;
            stopBtn.disabled = false;
            startBtn.classList.add("recording");
        } catch (err) {
            console.error("Error accessing microphone:", err);
            alert("Microphone access is required for recording. Please allow permission.");
        }
    });

    stopBtn.addEventListener("click", () => {
        if (activeRecorder) {
            activeRecorder.stop();
            activeRecorder = null;
            startBtn.disabled = false;
            stopBtn.disabled = true;
            startBtn.classList.remove("recording");
        }
    });
});

// Check Completion
function checkCompletion() {
    const complete = recordings.every((r) => r !== null);
    document.getElementById("submitVoice").disabled = !complete;
}

// Submit Handler
document.getElementById("submitVoice").addEventListener("click", () => {
    const formData = new FormData();
    recordings.forEach((blob, index) => {
        formData.append(`recording-${index}`, blob, `recording-${index}.wav`);
    });

    // Simulated submission
    fetch("/submit-recordings", {
        method: "POST",
        body: formData,
    })
        .then((response) => response.json())
        .then((data) => {
            alert("All recordings submitted successfully!");
        })
        .catch((error) => {
            console.error("Error:", error);
            alert("Failed to submit recordings. Please try again.");
        });
});