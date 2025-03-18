document.addEventListener("DOMContentLoaded", function () {
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

    let currentSentenceIndex = 0;
    let recordings = [];
    let mediaRecorder;
    let audioChunks = [];

    const sentenceText = document.getElementById("sentenceText");
    const currentSentence = document.getElementById("currentSentence");
    const startRecord = document.getElementById("startRecord");
    const stopRecord = document.getElementById("stopRecord");
    const audioPlayback = document.getElementById("audioPlayback");
    const nextSentence = document.getElementById("nextSentence");
    const prevSentence = document.getElementById("prevSentence");
    const submitVoice = document.getElementById("submitVoice");

    startRecord.addEventListener("click", async () => {
        let stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.start();
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            let audioBlob = new Blob(audioChunks, { type: "audio/wav" });
            let audioUrl = URL.createObjectURL(audioBlob);
            audioPlayback.src = audioUrl;
            audioPlayback.classList.remove("hidden");
            recordings[currentSentenceIndex] = audioBlob;
            nextSentence.disabled = false;
        };

        startRecord.disabled = true;
        stopRecord.disabled = false;
    });

    stopRecord.addEventListener("click", () => {
        mediaRecorder.stop();
        startRecord.disabled = false;
        stopRecord.disabled = true;
    });

    nextSentence.addEventListener("click", () => {
        if (currentSentenceIndex < sentences.length - 1) {
            currentSentenceIndex++;
            updateSentence();
        }
    });

    prevSentence.addEventListener("click", () => {
        if (currentSentenceIndex > 0) {
            currentSentenceIndex--;
            updateSentence();
        }
    });

    function updateSentence() {
        sentenceText.innerHTML = `<strong>"${sentences[currentSentenceIndex]}"</strong>`;
        currentSentence.innerText = currentSentenceIndex + 1;
        audioPlayback.classList.add("hidden");
        nextSentence.disabled = true;
        prevSentence.disabled = currentSentenceIndex === 0;
        submitVoice.classList.toggle("hidden", currentSentenceIndex !== sentences.length - 1);
    }

    submitVoice.addEventListener("click", () => {
        alert("Voice samples submitted successfully!");
    });
});
