document.addEventListener("DOMContentLoaded", function () {
    const loginPage = document.getElementById("loginPage");
    const enrollPage = document.getElementById("enrollPage");
    const loginForm = document.getElementById("loginForm");

    // Sample login event
    loginForm.addEventListener("submit", function (event) {
        event.preventDefault(); // Prevent actual submission
        loginPage.classList.add("hidden");
        enrollPage.classList.remove("hidden");
    });

    // Enrollment Page Logic
    const sentences = [
        "Hello, my name is [Name].", "I am [Name], present today.", "Yes, I am here.",
        "This is [Name] speaking.", "[Name] here, marking my attendance.", "Present and ready.",
        "Good morning, my name is [Name].", "How are you doing today?", "It’s a great day outside.",
        "Can you hear me clearly?", "I love learning new things.", "Let’s get started with today’s class.",
        "This is just a test sentence.", "Please repeat that once again.", "I hope everyone is doing well.",
        "The quick brown fox jumps over the lazy dog.", "I enjoy listening to music in my free time.",
        "Artificial intelligence is changing the world.", "A rolling stone gathers no moss.",
        "Technology is advancing at a rapid pace.", "Speech recognition is an interesting field of study.",
        "She sells seashells by the seashore.", "A journey of a thousand miles begins with a single step.",
        "My phone number is 9876543210.", "The time now is 10:30 AM.", "Today’s date is the 5th of March.",
        "I will be 22 years old next year.", "Yes.", "No.", "Okay.", "Hmm.", "Alright.", "Thank you."
    ];

    const sentencesContainer = document.getElementById("sentencesContainer");
    const submitVoice = document.getElementById("submitVoice");
    let recordings = new Array(sentences.length).fill(null);
    let recorders = new Array(sentences.length);
    let audioChunks = new Array(sentences.length);

    // Generate sentences UI
    sentences.forEach((sentence, index) => {
        const sentenceBlock = document.createElement("div");
        sentenceBlock.className = "sentence-block";

        sentenceBlock.innerHTML = `
            <p><strong>${sentence}</strong></p>
            <button class="startRecord" data-index="${index}">🎤 Start</button>
            <button class="stopRecord hidden" data-index="${index}">⏹️ Stop</button>
            <audio class="audioPlayback hidden" controls></audio>
        `;

        sentencesContainer.appendChild(sentenceBlock);
    });

    // Handle recording
    document.querySelectorAll(".startRecord").forEach((button) => {
        button.addEventListener("click", async function () {
            const index = parseInt(this.getAttribute("data-index"));
            let stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            recorders[index] = new MediaRecorder(stream);
            audioChunks[index] = [];

            recorders[index].start();
            recorders[index].ondataavailable = (event) => audioChunks[index].push(event.data);
            recorders[index].onstop = () => {
                let audioBlob = new Blob(audioChunks[index], { type: "audio/wav" });
                let audioUrl = URL.createObjectURL(audioBlob);
                document.querySelectorAll(".audioPlayback")[index].src = audioUrl;
                document.querySelectorAll(".audioPlayback")[index].classList.remove("hidden");
                recordings[index] = audioBlob;
                checkAllRecorded();
            };

            this.nextElementSibling.classList.remove("hidden");
            this.classList.add("hidden");
        });
    });

    function checkAllRecorded() {
        if (recordings.every((r) => r !== null)) {
            submitVoice.disabled = false;
        }
    }

    submitVoice.addEventListener("click", () => {
        alert("Voice samples submitted successfully!");
    });
});
