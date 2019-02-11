//Get camera video
const constraints = {
    audio: false,
    video: {
        facingMode: 'environment',
        width: {min: 320, ideal: 1280, max: 1920},
    }
};

navigator.mediaDevices.getUserMedia(constraints)
    .then(stream => {
        document.getElementById("camera").srcObject = stream;
        console.log("Got local user video");

    })
    .catch(err => {
        console.log('navigator.getUserMedia error: ', err)
    });
