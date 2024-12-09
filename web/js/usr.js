window.onload= function (){
    cookie = document.cookie
    function getCookieValue(name){
        let i =0;
        let cookieArr = cookie.split(';')
        for(i=0;i<cookieArr.length;i++){
            let cookieset = cookieArr[i].split("=")
            if (cookieset[0].trim() === name){
                return cookieset[1]
            }
        }
        return null
    }
    usrname = getCookieValue("usrnm")
    usrcity = getCookieValue("city")
    if(cookie==="" || usrname===null){
        alert("登录信息失效！")
        window.location.href="usr_mng.html"
    }
    usrnm = document.getElementById("usrnm")
    usrcty = document.getElementById("booked")
    usrnm.innerHTML = usrname
    if(usrcity!==null){
        usrcty.innerHTML = usrcity
    }
}
function deleteCookie(cookieName) {
    // 设置cookie的过期时间为过去的时间
    document.cookie = cookieName + '=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/';
}
function logout(){
    deleteCookie("usrnm")
    window.location.href="usr_mng.html"
}