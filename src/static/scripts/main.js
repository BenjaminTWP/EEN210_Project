// OPEN AN CLOSE THE ACCORDION WITH DATA COLLECTION
const accordion = document.querySelector('.accordion');
accordion.addEventListener('click', function() {
    this.classList.toggle('active');

    const panel = this.nextElementSibling;
    
    if (panel.style.display === "block") {
        panel.style.display = "none";
    } else {
        panel.style.display = "block";
    }
});

// SETUP CHARTS
var accelChart = new Chart(document.getElementById('accelChart').getContext('2d'), {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            { label: 'Acceleration X', data: [], borderColor: 'red' },
            { label: 'Acceleration Y', data: [], borderColor: 'green' },
            { label: 'Acceleration Z', data: [], borderColor: 'blue' }
        ]
    }
});

var gyroChart = new Chart(document.getElementById('gyroChart').getContext('2d'), {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            { label: 'Gyroscope X', data: [], borderColor: 'purple' },
            { label: 'Gyroscope Y', data: [], borderColor: 'orange' },
            { label: 'Gyroscope Z', data: [], borderColor: 'cyan' }
        ]
    }
});


// CONNECT WEBSOCKET AND HANDLE MESSAGES

var recordData = false;

var data = null;

var ipAddress = "192.168.75.220";

var ws = new WebSocket("ws://" + ipAddress + ":8000/ws"); // Create a new WebSocket

ws.onopen = function (event) {
    console.log("WebSocket state:", ws.readyState);  // This will log "OPEN"
};

var fallDetected = false;

ws.onmessage = function (event) {
    //console.log("Received data:", event.data);
    var message = JSON.parse(event.data);

    if (message.type === "sensor") {
        updateLiveCharts(message.data);
        if (recordData) {
            updateRecordingCharts(message.data);
        }
    } 
    else if (message.type === "prediction") {
        console.log("Prediction received:", message.label);
        updatePredictionLabel(message.label);
    }
    else if (message.type === "patient_info") {
        console.log("Patient Info Received:", message);
        updatePatientInfo(message);
    }
};

function updatePredictionLabel(prediction) {
    const labels = {
        fall: document.getElementById("fallLabel"),
        still: document.getElementById("stillLabel"),
        other: document.getElementById("otherLabel")
    };

    Object.values(labels).forEach(label => {
        label.style.backgroundColor = "transparent";
        label.style.color = "black"; 
    });

    const labelColors = {
        fall: { background: "red", text: "white" },
        still: { background: "green", text: "white" },
        other: { background: "yellow", text: "black" }
    };

    if (labels[prediction]) {
        labels[prediction].style.backgroundColor = labelColors[prediction].background;
        labels[prediction].style.color = labelColors[prediction].text;
    }
}

function updatePatientInfo(info) {
    document.getElementById("patientName").innerText = info.name;
    document.getElementById("patientID").innerText = info.id;
    document.getElementById("patientAge").innerText = info.age;
    document.getElementById("medList").innerText = info.medications.join(", ") || "None";
    document.getElementById("careList").innerText = info.careplans.join(", ") || "None";
    document.getElementById("contactList").innerText = info.contact;
}


function updateRecordingCharts(data) {
    // This is the label of the time index. Right now nothing. 
    accelChart.data.labels.push("");
    gyroChart.data.labels.push("");

    accelChart.data.datasets[0].data.push(data.acceleration_x);
    accelChart.data.datasets[1].data.push(data.acceleration_y);
    accelChart.data.datasets[2].data.push(data.acceleration_z);

    gyroChart.data.datasets[0].data.push(data.gyroscope_x);
    gyroChart.data.datasets[1].data.push(data.gyroscope_y);
    gyroChart.data.datasets[2].data.push(data.gyroscope_z);

    gyroChart.update();
}

close = function () {
    // ws.close();
    console.log("Connection Closed");
};

// Add onclose event handler
ws.onclose = function (event) {
    console.log("WebSocket closed:", event);
};
            

function clearData(){
    gyroChart.data.labels = [];  
    gyroChart.data.datasets.forEach(dataset => {
        dataset.data = []; 
    });

    accelChart.data.labels = [];  
    accelChart.data.datasets.forEach(dataset => {
        dataset.data = []; 
    });

    accelChart.update();
    gyroChart.update(); 
}

// START, CLEAR AND STOP BUTTON
var startButton = document.getElementById("startButton");
startButton.addEventListener("click", function () {
    if (!recordData){
        clearData();
        startButton.style.backgroundColor = "red";
        stopButton.style.backgroundColor = "greenyellow"
        clearButton.style.backgroundColor = "red"
        recordData = true;
        let filename = document.getElementById("filename").value || "data_log";
        ws.send(JSON.stringify({ action: "start", filename: filename }));
        console.log("Recording started with filename:", filename);
    }
});

var stopButton = document.getElementById("stopButton");
stopButton.addEventListener("click", function () {
    if (recordData){
        startButton.style.backgroundColor = "greenyellow";
        stopButton.style.backgroundColor = "red";
        clearButton.style.backgroundColor = "greenyellow";
        recordData = false;
        ws.send(JSON.stringify({ action: "stop" }));
        console.log("Recording stopped.");
    }
});

var clearButton = document.getElementById("clearButton");
clearButton.addEventListener("click", function (){
    if (!recordData){
        clearData();
    }
    var ul = document.getElementById("messages");
    
});

function sendEmailCall() {
    let filename = document.getElementById("filename").value || "data_log";
    console.log("Sending WebSocket message for email with filename:", filename);
    
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: "email", filename: filename }));
    } else {
        console.error("WebSocket is NOT open! Current state:", ws.readyState);
    }
}
document.getElementById("sendButton").addEventListener("click", sendEmailCall);