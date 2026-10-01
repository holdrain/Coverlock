const copyButton = document.querySelector("#copy-bibtex");

copyButton?.addEventListener("click", async () => {
  const citation = document.querySelector("#bibtex code")?.textContent ?? "";
  try {
    await navigator.clipboard.writeText(citation);
    copyButton.textContent = "Copied";
    window.setTimeout(() => { copyButton.textContent = "Copy BibTeX"; }, 1600);
  } catch {
    copyButton.textContent = "Select to copy";
  }
});
