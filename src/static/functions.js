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


function emptyPostRequest(url) {
    fetch(url, {
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

function postRequest(url, payload) {
    fetch(url, {
                method: 'post',
                headers: {
                    'Accept': 'application/json, text/plain, */*',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
             }
         )
        .then((res) => {return res.json()})
        .then((data) => {
            console.log(data);
        })
}


function postRequestForm(url, formElement) {
    let payload = {};
    $.each($(formElement).serializeArray(), function() {
            payload[this.name] = this.value;
        }
    );
    let formData = JSON.stringify(payload);

    fetch(url, {
                method: 'post',
                headers: {
                    'Accept': 'application/json, text/plain, */*',
                    'Content-Type': 'application/json'
                },
                body: formData
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