def create_audio_recorder(default_filename="your_voice.wav", width="320px",
                          background_color="#f9f9f9", auto_install=True):
    """
    Create an interactive audio recorder widget (WAV, 16-bit PCM, mono) for Jupyter notebooks.

    Recording runs in the browser via an AudioWorklet, is encoded to WAV there, and the raw
    bytes are sent to the kernel over the standard widget channel (anywidget), where the file
    is saved. This gives ONE code path for all environments: Google Colab, VS Code,
    JupyterLab (>= 3) and Jupyter Notebook (>= 7).

    Everything (imports, dependency check, widget class) lives inside this function, so it can
    be copied into a library or a notebook cell as-is.

    Parameters
    ----------
    default_filename : str
        Suggested file name (relative to the kernel's working directory, or absolute).
        Missing subfolders are created; ".wav" is appended if missing.
    width : str
        Maximum widget width as a CSS value, e.g. "320px" or "100%".
    background_color : str
        Background color as a CSS value.
    auto_install : bool
        If True and `anywidget` is not installed, install it via pip into the kernel's
        environment. If False, raise an ImportError with installation instructions instead.

    Returns
    -------
    AudioRecorder (anywidget.AnyWidget)
        Displayed when it is the last expression of a cell (or via
        `IPython.display.display(rec)`). After recording, `rec.path` holds the pathlib.Path
        of the saved file (None until something was saved) and `rec.status` the status text.

    Requirements
    ------------
    * anywidget (installed automatically if `auto_install=True`; otherwise
      `pip install anywidget`, or `!pip install -q anywidget` in Colab)
    * Locally: ipykernel in the kernel's environment; in VS Code the "Jupyter" extension.

    Usage
    -----
        from audio_recorder import create_audio_recorder   # or paste the function into a cell

        rec = create_audio_recorder("recordings/test.wav")
        rec                               # last line of the cell -> widget is shown

        # after Start/Stop, in a later cell:
        from IPython.display import Audio
        Audio(rec.path)

    Notes
    -----
    * The browser (or VS Code) asks for microphone permission on first start. getUserMedia
      only works in a secure context (https or localhost).
    * The file is saved where the kernel runs: the Colab VM in Colab, your machine locally.
      The "download locally" checkbox is therefore mainly useful in Colab.
    * Multiple recorders in the same notebook do not interfere with each other.
    """
    import importlib
    import subprocess
    import sys
    from pathlib import Path

    # --- Dependency handling -----------------------------------------------------------
    try:
        import anywidget
    except ImportError:
        if not auto_install:
            raise ImportError(
                "create_audio_recorder needs 'anywidget'. Install it with "
                "'pip install anywidget' (in Colab: '!pip install -q anywidget')."
            ) from None
        print("Installing 'anywidget' ...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "anywidget"])
        except subprocess.CalledProcessError:
            # e.g. a protected system Python (PEP 668) or no network access
            raise ImportError(
                "Automatic installation of 'anywidget' failed. Install it manually into the "
                f"kernel's environment ({sys.executable}), ideally inside a virtual environment."
            ) from None
        importlib.invalidate_caches()
        import anywidget
    import traitlets  # installed together with anywidget (via ipywidgets)

    if "google.colab" in sys.modules:
        from google.colab import output
        output.enable_custom_widget_manager()  # Colab needs this for third-party widgets

    # --- Runs on the browser's audio thread --------------------------------------------
    # Collects the 128-sample blocks into larger packets (fewer messages to the main thread)
    # and sends them over the MessagePort. On "flush" it sends the remainder and signals the end.
    worklet_js = r"""
    class RecorderProcessor extends AudioWorkletProcessor {
      constructor() {
        super();
        this.buf = new Float32Array(4096);
        this.len = 0;
        this.port.onmessage = (e) => {
          if (e.data === "flush") {
            if (this.len) this.port.postMessage(this.buf.slice(0, this.len));
            this.len = 0;
            this.port.postMessage("done");
          }
        };
      }
      process(inputs) {
        const ch = inputs[0] && inputs[0][0];        // first input, first channel (mono)
        if (ch) {
          for (let i = 0; i < ch.length; i++) {
            this.buf[this.len++] = ch[i];
            if (this.len === this.buf.length) {
              const full = this.buf;
              this.port.postMessage(full, [full.buffer]);   // transfer instead of copy
              this.buf = new Float32Array(4096);
              this.len = 0;
            }
          }
        }
        return true;                                  // keep the processor alive
      }
    }
    registerProcessor("recorder-processor", RecorderProcessor);
    """

    # --- Runs on the main thread (widget UI) -------------------------------------------
    widget_js = r"""
    function encodeWAV(chunks, sampleRate) {
      const n = chunks.reduce((sum, c) => sum + c.length, 0);
      const view = new DataView(new ArrayBuffer(44 + n * 2));
      const str = (off, s) => [...s].forEach((ch, i) => view.setUint8(off + i, ch.charCodeAt(0)));
      str(0, "RIFF"); view.setUint32(4, 36 + n * 2, true); str(8, "WAVE");
      str(12, "fmt "); view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);              // PCM
      view.setUint16(22, 1, true);              // mono
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true); // byte rate
      view.setUint16(32, 2, true);              // block align
      view.setUint16(34, 16, true);             // bits per sample
      str(36, "data"); view.setUint32(40, n * 2, true);
      let off = 44;
      for (const c of chunks) for (const x of c) {
        const s = Math.max(-1, Math.min(1, x));
        view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        off += 2;
      }
      return view.buffer;
    }

    function render({ model, el }) {
      el.innerHTML = `
        <div class="rec">
          <label>File name:<input class="name" type="text"></label>
          <label class="small"><input class="dl" type="checkbox"> Also download locally</label>
          <div>
            <button class="start">🎙️ Start</button>
            <button class="stop" disabled>🛑 Stop</button>
          </div>
          <div class="status"></div>
          <audio class="player" controls hidden></audio>
        </div>
        <style>
          .rec { padding:15px; border:1px solid #ccc; border-radius:8px; font-family:sans-serif;
                 display:flex; flex-direction:column; gap:10px; color:#333; }
          .rec label { font-size:13px; font-weight:bold; display:flex; flex-direction:column; gap:4px; }
          .rec label.small { flex-direction:row; align-items:center; font-weight:normal; font-size:12px; }
          .rec input[type=text] { padding:6px; border:1px solid #ccc; border-radius:4px; }
          .rec button { color:white; border:none; padding:10px 15px; border-radius:4px;
                        font-weight:bold; cursor:pointer; margin-right:5px; }
          .rec button:disabled { opacity:.5; cursor:default; }
          .rec .start { background:#28a745; } .rec .stop { background:#dc3545; }
          .rec .status { font-size:14px; }
        </style>`;

      const $ = (sel) => el.querySelector(sel);   // search only inside THIS widget
      const box = $(".rec"), nameInput = $(".name"), dl = $(".dl");
      const startBtn = $(".start"), stopBtn = $(".stop"), status = $(".status"), player = $(".player");

      box.style.maxWidth = model.get("width");
      box.style.background = model.get("background_color");
      nameInput.value = model.get("filename");

      const showStatus = () => (status.textContent = model.get("status"));
      model.on("change:status", showStatus);
      showStatus();

      const setRecording = (on) => {
        startBtn.disabled = on; stopBtn.disabled = !on;
        nameInput.disabled = on; dl.disabled = on;
      };

      let ctx = null, stream = null, node = null, chunks = [], finished = null;

      startBtn.onclick = async () => {
        startBtn.disabled = true;
        status.textContent = "Requesting microphone permission...";
        try {
          stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          ctx = new AudioContext();

          // Load the worklet code from the string (no separate file hosting needed)
          const url = URL.createObjectURL(new Blob([model.get("_worklet")], { type: "text/javascript" }));
          await ctx.audioWorklet.addModule(url);
          URL.revokeObjectURL(url);

          node = new AudioWorkletNode(ctx, "recorder-processor");
          chunks = [];
          finished = new Promise((resolve) => {
            node.port.onmessage = (e) => (e.data === "done" ? resolve() : chunks.push(e.data));
          });

          // Connect to destination so the graph is reliably processed.
          // The processor writes nothing to its output -> nothing is audible.
          ctx.createMediaStreamSource(stream).connect(node).connect(ctx.destination);
          await ctx.resume();

          setRecording(true);
          status.textContent = "🔴 Recording...";
        } catch (err) {
          if (stream) stream.getTracks().forEach((t) => t.stop());
          if (ctx) ctx.close();
          startBtn.disabled = false;
          status.textContent = "Error: " + err.message;
        }
      };

      stopBtn.onclick = async () => {
        stopBtn.disabled = true;
        stream.getTracks().forEach((t) => t.stop());
        node.port.postMessage("flush");          // fetch the remaining buffer from the worklet
        await finished;
        const sampleRate = ctx.sampleRate;
        await ctx.close();
        setRecording(false);

        if (!chunks.length) {
          status.textContent = "Error: No audio data recorded!";
          return;
        }

        let name = nameInput.value.trim() || "recording.wav";
        if (!/\.wav$/i.test(name)) name += ".wav";
        nameInput.value = name;

        const wav = encodeWAV(chunks, sampleRate);
        const url = URL.createObjectURL(new Blob([wav], { type: "audio/wav" }));
        player.src = url;
        player.hidden = false;

        if (dl.checked) {
          const a = Object.assign(document.createElement("a"), { href: url, download: name });
          el.appendChild(a); a.click(); a.remove();
        }

        status.textContent = "Sending to kernel...";
        model.send({ filename: name }, undefined, [wav]);   // raw bytes, no base64 needed
      };
    }

    export default { render };
    """

    # --- Python side of the widget -----------------------------------------------------
    class AudioRecorder(anywidget.AnyWidget):
        _esm = widget_js
        _worklet = traitlets.Unicode(worklet_js).tag(sync=True)

        filename = traitlets.Unicode("your_voice.wav").tag(sync=True)
        width = traitlets.Unicode("320px").tag(sync=True)
        background_color = traitlets.Unicode("#f9f9f9").tag(sync=True)
        status = traitlets.Unicode("Ready to record...").tag(sync=True)

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.path = None  # pathlib.Path of the last saved recording
            self.on_msg(self._on_audio)

        def _on_audio(self, _widget, content, buffers):
            try:
                path = Path(content["filename"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(bytes(buffers[0]))
                self.path = path
                self.status = f"💾 Saved: {path.resolve()}"
            except Exception as exc:  # show errors in the widget instead of losing them in the log
                self.status = f"Error while saving: {exc}"

    return AudioRecorder(filename=default_filename, width=width, background_color=background_color)
