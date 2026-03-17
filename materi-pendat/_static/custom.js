document.addEventListener("DOMContentLoaded", function() {
    let copyRightElement = document.querySelector(".copyright");
    if (copyRightElement) {
        let currentYear = new Date().getFullYear();
        copyRightElement.innerHTML = `© Copyright ${currentYear}, ShofiatulMahmudah. <br/>`;
    }
});
