function uploadImage() {
    const highlightedElements = document.querySelectorAll('.highlight');
    highlightedElements.forEach(element => {
        element.remove();
    });
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];

    const formData = new FormData();
    formData.append('image', file);

    fetch('/annotate', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const annotatedImage = document.getElementById('annotatedImage');
        annotatedImage.src = 'data:image/png;base64,' + data.image;

        const boundingBoxInfo = document.getElementById('boundingBoxInfo');
        boundingBoxInfo.innerHTML = '';

        for (const boxId in data.bounding_boxes) {
            const boxInfo = data.bounding_boxes[boxId];
            const modeColor = data.mode_colors[boxId];

            const boxElement = document.createElement('div');
            boxElement.className = 'colorBox'
            boxElement.textContent = `rgb(${modeColor[0]}, ${modeColor[1]}, ${modeColor[2]})`;
            boxElement.style.backgroundColor = `rgb(${modeColor[0]}, ${modeColor[1]}, ${modeColor[2]})`;
            boxElement.onclick = () => highlightBoundingBox(boxInfo, modeColor);

            boundingBoxInfo.appendChild(boxElement);
        }
    })
    .catch(error => console.error('Error:', error));
}

function highlightBoundingBox(boxInfo) {
    const highlightedElements = document.querySelectorAll('.highlight');
    highlightedElements.forEach(element => {
        element.remove();
    });
    const annotatedImage = document.getElementById('annotatedImage');
    const imageRect = annotatedImage.getBoundingClientRect();

    const scaleX = annotatedImage.width / annotatedImage.naturalWidth;
    const scaleY = annotatedImage.height / annotatedImage.naturalHeight;

    const x = boxInfo[0] * scaleX + imageRect.x;
    const y = boxInfo[1] * scaleY + imageRect.y;
    const w = boxInfo[2] * scaleX - 9;
    const h = boxInfo[3] * scaleY - 9;

    const boundingBox = document.createElement('div');
    boundingBox.className = 'highlight';
    boundingBox.style.left = `${x}px`;
    boundingBox.style.top = `${y}px`;
    boundingBox.style.width = `${w}px`;
    boundingBox.style.height = `${h}px`;

    document.body.appendChild(boundingBox);
}