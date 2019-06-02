// Image Preprocessing
// Resize and apply normalization to training data
const jimp = require('jimp');

var trainingSize = 500;

module.exports.process = function() {
    preprocessAll();
}




async function preprocessPhoto(c, t_id) {
    let imgRaw = 'training_data/exp4/raw/class_' + c + '/training_' + t_id + '.png';
    let imgActive = 'training_data/exp4/active/class_' + c + '/training_' + t_id + '.png';
    let imgExported = 'training_data/exp4/export/class_' + c + '/training_' + t_id + '.png';

    await jimp.read(imgRaw)
    .then(img => (img.clone().write(imgActive)))
    .then(() => jimp.read(imgActive))
    .then(img => {return img.crop(370,230,470,280);})
    // .then(img => {return img.resize(100,100);})
    // .then(img => {return img.crop(20,35,60,45);})
    .then(img => img.resize(28,28))
    // .then(img => img.normalize())
    // .then(img => img.grayscale())
    // .then(img => img.write(imgExported))
    .then(() => console.log('Exported file to: ' + imgExported))
    .catch((err) => {console.error(err); preprocessPhoto(c, t_id)});
    console.log(c+': '+t_id);    
}

async function preprocessPhotos(c) {
    var t_id = 0;
    while(t_id < trainingSize) {
        await preprocessPhoto(c, t_id);
        ++t_id;
    }
}


async function preprocessAll() {
    await preprocessPhotos(0);
    await preprocessPhotos(1);
    await preprocessPhotos(2);
    await preprocessPhotos(3);
    await preprocessPhotos(4);
    await preprocessPhotos(5);
    await preprocessPhotos(6);
    await preprocessPhotos(7);
    await preprocessPhotos(8);
    await preprocessPhotos(9);
    await preprocessPhotos(10);
    await preprocessPhotos('1-3');
    await preprocessPhotos('1-3-10');
    await preprocessPhotos('1-10');
    await preprocessPhotos('2-6-10');
    await preprocessPhotos('3-4-5');
    await preprocessPhotos('3-4-5-11');
    await preprocessPhotos('3-10');
    await preprocessPhotos('4-5');
    await preprocessPhotos('5-10');
    await preprocessPhotos('6-10');
    console.log("Complete");
}