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

function toggleOn() {
    $('#status-button').bootstrapToggle('enable').bootstrapToggle('on').bootstrapToggle('disable')
}
function toggleOff() {
    $('#status-button').bootstrapToggle('enable').bootstrapToggle('off').bootstrapToggle('disable')
}