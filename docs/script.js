const demo = document.querySelector("#transfer-demo");
const forgeButton = document.querySelector("#forge-button");
const copyButton = document.querySelector("#copy-bibtex");

forgeButton?.addEventListener("click", () => {
  demo.classList.toggle("is-forged");
});

copyButton?.addEventListener("click", async () => {
  const citation = document.querySelector("#bibtex code")?.textContent ?? "";
  try {
    await navigator.clipboard.writeText(citation);
    copyButton.textContent = "Copied";
    window.setTimeout(() => {
      copyButton.textContent = "Copy citation";
    }, 1800);
  } catch {
    copyButton.textContent = "Select and copy";
  }
});

const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) entry.target.classList.add("is-visible");
  });
}, { threshold: 0.16 });

document.querySelectorAll(".finding-grid article, .asr-grid article, .method-facts div")
  .forEach((element) => observer.observe(element));
