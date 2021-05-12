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

 function getJsonList(url){
    return fetch(url,
    {
    	method: "GET",
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      },
    })
    .then((response) => response.json())
    .then((responseData) => {
      return responseData;
    })
    .catch(error => console.warn(error));
  }


  function populateDropDown(url, dropdownList) {

    getJsonList(url)
    .then(response => {
        console.log(response)
        var list = document.getElementById(dropdownList);
        for (var i = 0; i < response.length; i++){
            var opt = response[i];
            var li = document.createElement("li");
            var link = document.createElement("a");
            var text = document.createTextNode(opt);
            link.appendChild(text);
            link.href = "#";
            li.appendChild(link);
            list.appendChild(li);
        }

    });




}

function toggleOn() {
    $('#status-button').bootstrapToggle('enable').bootstrapToggle('on').bootstrapToggle('disable')
}
function toggleOff() {
    $('#status-button').bootstrapToggle('enable').bootstrapToggle('off').bootstrapToggle('disable')
}