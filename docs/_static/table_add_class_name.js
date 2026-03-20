// 在页面加载完成后执行
document.addEventListener('DOMContentLoaded', function () {

    // 选择非Sphinx创建的表格元素
    const tables = document.querySelectorAll('table:not([class*="docutils"])');

    tables.forEach(table => {
        // 给 <table> 添加 Sphinx 默认 class
        table.classList.add('docutils', 'align-default');

        table.querySelectorAll('th').forEach(th => {
            th.classList.add('head');

        });

        const row = table.querySelectorAll('tr');
        for (let i = 0; i < row.length; i++) {
            row[i].classList.add(i % 2 === 0 ? 'row-odd' : 'row-even')
        }

        table.querySelectorAll('td[rowspan]').forEach(cell => {
            cell.style.borderRight = '1px solid #e1e4e5';
        });
    });
});
