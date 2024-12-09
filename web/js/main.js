import DPlayer from '../node_modules/dplayer/dist/DPlayer.min.js';
function getCookieValue(name){
    let cookieArr = document.cookie.split(';')
    let i;
    for(i=0;i<cookieArr.length;i++){
        let cookieset = cookieArr[i].split("=")
        if (cookieset[0].trim() === name){
            return cookieset[1]
        }
    }
    return null
}

const dp = new DPlayer({
    container: document.getElementById('dplayer'),
    video: {
        url: 'image/intro.mp4',
    },
});

// const container = document.querySelector('.container');
$(document).ready(function (){
    let welcome_Page = $('.welcome_page');
    let nav = $('.p1_nav')
    let welcomeImglst = ['image/case01.jpg','image/case02.jpg','image/case03.jpg','image/case04.jpg','image/case05.jpg']
    let chg_index = 0;
    let usr_btn = $('.login-btn');
    let usr_hint = $('.logspan')
    let container = $('.container');
    let lastScrollTop = 0; //上一次滚动
    if(getCookieValue("usrnm")!==null){
        usr_hint.text("欢迎"+getCookieValue("usrnm")+"!")
        usr_btn.val("用户中心")
    }else{
        usr_hint.text("您未登录")
        usr_btn.val("登录")
    }
    var changeImg = function (){
        chg_index+=1
        if(chg_index>4){
            chg_index = 1
        }
        $('.welcome_page').css('background-image', "url("+welcomeImglst[chg_index]+")"); // 更改图片的src属性
    }
    setInterval(changeImg,3000);
    $('#exp').click(function (){
        window.location.href="explore.html"
    })
    $('#scenery').click(function (){
        window.location.href="/body"
    })
    // 橱窗翻页
    $(window).scroll(function (){
        let curr_Scroll = $(this).scrollTop();
        if(curr_Scroll > lastScrollTop){
            //向下
            if(curr_Scroll>10){
                welcome_Page.css("top",'-100vh')
                container.css('top','0vh')
                nav.css("background-color","rgba(242,242,242,1)")

            }
        }else{
            //向上
            if(curr_Scroll< 50){
                welcome_Page.css("top",'0')
                container.css('top','100vh')
                nav.css("background-color","rgba(255,255,255,0)")
            }
        }
        lastScrollTop = curr_Scroll
    })
})