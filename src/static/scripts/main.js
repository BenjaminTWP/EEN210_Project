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

var ipAddress = "192.168.27.220";

var ws = new WebSocket("ws://" + ipAddress + ":8000/ws"); // Create a new WebSocket
console.log("we do get here")
ws.onopen = function (event) {
    console.log("WebSocket state:", ws.readyState);  // This will log "OPEN"
};
ws.onmessage = function (event) {
    console.log("Received data:", event.data);
    var messages = document.getElementById('messages');
    var message = document.createElement('li');
    var content = document.createTextNode(event.data);
    message.appendChild(content);

    // Append the new li element to the ul element
    messages.appendChild(message);

    // Scroll to the bottom
    messages.scrollTop = messages.scrollHeight;

    var data = JSON.parse(event.data);
    if(recordData){
        updateCharts(data);
    }
};


function updateCharts(data) {
    // This is the label of the time index. Right now nothing. 
    accelChart.data.labels.push("");
    gyroChart.data.labels.push("");

    accelChart.data.datasets[0].data.push(data.acceleration_x);
    accelChart.data.datasets[1].data.push(data.acceleration_y);
    accelChart.data.datasets[2].data.push(data.acceleration_z);

    gyroChart.data.datasets[0].data.push(data.gyroscope_x);
    gyroChart.data.datasets[1].data.push(data.gyroscope_y);
    gyroChart.data.datasets[2].data.push(data.gyroscope_z);

    accelChart.update();
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
            
var closeButton = document.getElementById('closeButton');
    closeButton.addEventListener('click', function () {
    // Call the close function when the button is clicked
    close();
});

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
});