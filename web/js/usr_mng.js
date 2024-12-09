function login() {
    // 获取表单数据
    var username = document.getElementById('username').value;
    var password = document.getElementById('password').value;
    let str = localStorage.getItem(username)
    let data = JSON.parse(str);


    if (!data)
        alert("没有注册，请注册");
    if (data.password == password){
        alert("登陆成功，即将跳转至个人信息页")
        document.cookie="usrnm="+username + ";path=/";
        window.location.href="../usr/usr_info.html"}
    else
        alert("密码不正确")
    
    // x表单提交
    return false;
}
function register() {
    // 获取表单数据
    let data = new Object
    data.username = document.getElementById('regUsername').value;
    var email = document.getElementById('regEmail').value;
    data.password = document.getElementById('regPassword').value;
    var confirmPassword = document.getElementById('regConfirmPassword').value;

    // 确认两次输入的密码是否一致
    if(data.password !== confirmPassword) {
        alert('两次输入的密码不一致，请重新输入！');
        return false;
    }

    let str = JSON.stringify(data);
    localStorage.setItem(data.username,str);
    alert("数据已保存。");

    // 防止表单提交
    return false;
}
