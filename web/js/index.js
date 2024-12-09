document.getElementById('next').onclick = function() {
    let lists = document.querySelectorAll('.box');
    document.getElementById('slide').appendChild(lists[0]);
}

document.getElementById('previous').onclick = function() {
    let lists = document.querySelectorAll('.box');
    let lastItem = lists[lists.length - 1];
    document.getElementById('slide').insertBefore(lastItem, lists[0]);
}

window.CSS.registerProperty({
    name:'--primaryColor',
    syntax:'<color>',
    inherits:false,
    initialValue:'#aa00ff',
});

window.CSS.registerProperty({
    name:'--secondaryColor',
    syntax:'<color>',
    inherits:false,
    initialValue:'#ff2661',
});

const curtainToggle = document.querySelector('.curtain-toggle');
  const curtain = document.querySelector('.curtain');
  const curtain1 = document.querySelector('.curtain1');

  curtainToggle.onclick = function() {
    curtain.classList.add('open');
    curtain1.classList.add('open');

    curtainToggle.style.display = 'none'; // 将按钮隐藏
  };