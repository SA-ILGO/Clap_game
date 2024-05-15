function animateFirstImage() {
  var firstImage = document.getElementById("up");
  firstImage.addEventListener("animationend", function() {
      animateSecondImage();
  });
}

function animateSecondImage() {
  var secondImage = document.getElementById("up1");
  // 두 번째 이미지의 애니메이션을 시작합니다.
  secondImage.style.animation = "moveLeft 1s linear forwards";
}

animateFirstImage();
