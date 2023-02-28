function request_classify(){
    const form = document.getElementById('uploadForm');
    const fileInput = document.getElementById('image_file');
    const responseDiv = document.getElementById('response');

    form.addEventListener('submit', (e) => {
        e.preventDefault();

        const formData = new FormData();
        formData.append('image', fileInput.files[0]);

        fetch('http://localhost:5001/read/inference/apple', {
            method: 'POST',
            body: formData,
            headers: {
                //specify which domains are allowed to make requests to server
                'Access-Control-Allow-Origin':'*', //allow all access for testing purposes
                //header to specify which HTTP methods are allowed
                'Access-Control-Allow-Methods': 'POST',
                //This header specifies which headers are allowd in the request
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        })
        .then(response => response.json())
        .then(data => {
            var stringData = JSON.stringify(data, null, 1)
            responseDiv.innerHTML = '<pre>' + stringData + '</pre>';
        })
        .catch(error => {
            responseDiv.innerHTML = 'Error uploading image';
            console.error(error);
        });
    });
}