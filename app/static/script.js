document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".token").forEach(el => {
        let score = parseFloat(el.dataset.score);
        if (isNaN(score)) score = 0;

        // Red intensity heatmap
        el.style.backgroundColor = `rgba(255, 0, 0, ${score})`;
    });
});
