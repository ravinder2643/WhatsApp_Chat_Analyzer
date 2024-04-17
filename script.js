const form = document.getElementById('form');
const text = document.getElementById('text');
const qrcode = document.getElementById('qrcode');

const generateQRCode = (text) => {
  qrcode.innerHTML = '';
  const qr = new QRCode(qrcode, {
    text: text,
    width: 200,
    height: 200,
  });
};

form.addEventListener('submit', (e) => {
  e.preventDefault();
  const textValue = text.value;
  if (textValue === '') {
    alert('Please enter some text.');
    return;
  }
  generateQRCode(textValue);
});

generateQRCode('Hello, world!');