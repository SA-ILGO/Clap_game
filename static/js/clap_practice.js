const egg = document.getElementById("egg");
const edge = document.getElementById("edge");
const fist = document.getElementById("fist");
const wrist = document.getElementById("wrist");

const clapL = [egg, edge, fist, wrist];

const filePath = 'static/js/clap_data_now.txt';

function watch_dataFile() {
    // setInterval stop
    var intervalId;
    function load_dataFile() {
        fetch(filePath)
            .then(response => response.text())
            .then(data => {
                var lines = data.split('\n');
                var clap = lines[lines.length - 2];

                try{
                    if(clap.trim() == "egg clap"){
                        egg.style.borderColor = "rgb(255, 0, 0)";
                        fist.style.borderColor = "rgb(255, 255, 255)";
                        edge.style.borderColor = "rgb(255, 255, 255)";
                        wrist.style.borderColor = "rgb(255, 255, 255)";
                    } else if(clap.trim() == "fist clap"){
                        egg.style.borderColor = "rgb(255, 255, 255)";
                        fist.style.borderColor = "rgb(255, 0, 0)";
                        edge.style.borderColor = "rgb(255, 255, 255)";
                        wrist.style.borderColor = "rgb(255, 255, 255)";
    
                    } else if(clap.trim() == "edge clap"){
                        egg.style.borderColor = "rgb(255, 255, 255)";
                        fist.style.borderColor = "rgb(255, 255, 255)";
                        edge.style.borderColor = "rgb(255, 0, 0)";
                        wrist.style.borderColor = "rgb(255, 255, 255)";
    
                    } else if(clap.trim() == "wrist clap") {
                        egg.style.borderColor = "rgb(255, 255, 255)";
                        fist.style.borderColor = "rgb(255, 255, 255)";
                        edge.style.borderColor = "rgb(255, 255, 255)";
                        wrist.style.borderColor = "rgb(255, 0, 0)";
    
                    } 
                }catch(error){
                        
                }
            })
            .catch(error => console.error('file load error', error));
    }
    intervalId = setInterval(load_dataFile, 500); // 0.5

    window.addEventListener('unload', () => {
        clearInterval(intervalId);
    });
  };

  window.addEventListener('load', () => {
    watch_dataFile();
});