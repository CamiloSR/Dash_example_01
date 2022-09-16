// JavaScript Document
setInterval(function myFunction() {
    var input, filter, divs, la, i, txtValue;
    input = document.getElementById("input-platforms");
    if (input.value == ""){
      document.getElementById("no-text-btn").style.display = "none";
    } else {
      document.getElementById("no-text-btn").style.position = "absolute";
      document.getElementById("no-text-btn").style.scale = "80%";
      document.getElementById("no-text-btn").style.left = "97%";
      document.getElementById("no-text-btn").style.display = "";
    }
    filter = input.value.toUpperCase();
    divs = document.getElementById("csr-platforms");
    la = divs.getElementsByTagName("label");

    for (i = 0; i < la.length; i++) {
      txtValue = la[i].textContent || la[i].innerText || la[i].innerHTML;
      if (txtValue.toUpperCase().indexOf(filter) > -1) {
        la[i].style.display = "block";
      } else {
        la[i].style.display = "none";
      }
    }
  }, 100);