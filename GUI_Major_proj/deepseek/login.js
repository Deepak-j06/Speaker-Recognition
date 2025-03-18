// Login Handler
document.getElementById("loginForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const username = document.getElementById("username").value;
    const password = document.getElementById("password").value;

    // Simulate login success
    if (username && password) {
        window.location.href = "enroll.html"; // Redirect to enrollment page
    } else {
        alert("Please enter both username and password.");
    }
});