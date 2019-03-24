function getAPI(){
    fetch('http://127.0.0.1:5000/xbox/test')
        .then((res) => {return res.json()})
        .then((data) => {
            console.log(data);
            let result = '<h2> local results are in! </h2>';
            result +=
            `<div>
                <h2> result: ${data.x} </h2>
            </div>`;
            document.getElementById('result').innerHTML = result;
        })
}


function getRobotStatus(){
    fetch('http://127.0.0.1:5000/robotstatus')
        .then((res) => {return res.json()})
        .then((data) => {
            console.log(data);
            if (data.status) {
                toggleOn()
            } else {
                toggleOff()
            }
        })
}


function turnCameraOn() {
    fetch('http://127.0.0.1:5000/startcamera', {
                method: 'post',
                headers: {
                    'Accept': 'application/json, text/plain, */*',
                    'Content-Type': 'application/json'
                },
                body: ""
             }
         )
        .then((res) => {return res.json()})
        .then((data) => {
            console.log(data);
        })
}


function turnCameraOff() {
    fetch('http://127.0.0.1:5000/stopcamera', {
                method: 'post',
                headers: {
                    'Accept': 'application/json, text/plain, */*',
                    'Content-Type': 'application/json'
                },
                body: ""
             }
         )
        .then((res) => {return res.json()})
        .then((data) => {
            console.log(data);
        })
}


function toggleOn() {
    $('#status-button').bootstrapToggle('enable').bootstrapToggle('on').bootstrapToggle('disable')
}
function toggleOff() {
    $('#status-button').bootstrapToggle('enable').bootstrapToggle('off').bootstrapToggle('disable')
}