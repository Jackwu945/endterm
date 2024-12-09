// 获取模态框元素
var modal = document.getElementById("myModal");

// 获取打开模态框的按钮元素
var btn = document.getElementById("myBtn");

// 获取关闭模态框的 <span> 元素
var span = document.getElementsByClassName("close")[0];

// 当用户点击按钮时打开模态框
btn.onclick = function() {
    modal.style.display = "block";
}

// 当用户点击 <span> (x), 关闭模态框
span.onclick = function() {
    modal.style.display = "none";
}

// 当用户点击模态框以外的地方，也关闭模态框
window.onclick = function(event) {
    if (event.target == modal) {
        modal.style.display = "none";
    }
}