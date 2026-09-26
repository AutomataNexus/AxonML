//! V4L2 Backend — Linux Camera Capture via Video4Linux2
//!
//! Linux-specific camera capture using V4L2 ioctls and memory-mapped I/O.
//! Defines C-compatible structs (`V4l2Capability`, `V4l2Format`, `V4l2Buffer`)
//! whose sizes are pinned to the kernel ABI, and ioctl requests (QUERYCAP,
//! S_FMT, REQBUFS, QUERYBUF, QBUF, DQBUF, STREAMON, STREAMOFF) whose codes
//! are derived from those struct sizes and typed to them. `MmapBuffer` wraps mmap'd kernel buffers with RAII
//! cleanup. `V4L2Backend` implements `CaptureBackend` with zero-copy frame capture
//! from /dev/videoN devices, supporting YUYV and MJPEG pixel formats. Uses 4
//! memory-mapped buffers for continuous streaming.
//!
//! # File
//! `crates/axonml-vision/src/camera/v4l2.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 16, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use super::{CaptureBackend, CaptureConfig, CaptureError, FrameBuffer, PixelFormat};
use std::fs;
use std::os::unix::io::AsRawFd;

// =============================================================================
// V4L2 Constants
// =============================================================================

// V4L2 pixel format FourCC codes
const V4L2_PIX_FMT_YUYV: u32 = 0x5659_5559; // 'YUYV'
const V4L2_PIX_FMT_MJPEG: u32 = 0x4745_504D; // 'MJPG'

// V4L2 buffer types
const V4L2_BUF_TYPE_VIDEO_CAPTURE: u32 = 1;

// V4L2 memory types
const V4L2_MEMORY_MMAP: u32 = 1;

const NUM_BUFFERS: u32 = 4;

// =============================================================================
// ioctl request codes — derived from the argument type, never hardcoded
// =============================================================================

// Linux `_IOC(dir, type, nr, size)`: the kernel copies exactly `size` bytes
// to/from the user pointer, so `size` MUST equal the Rust struct's size. It
// is computed from `size_of::<Arg>()` below, which makes a code/struct size
// mismatch unrepresentable (the old hardcoded 32-bit codes described a
// 204-byte `v4l2_format` while the Rust struct was 180 bytes).
const IOC_WRITE: libc::c_ulong = 1;
const IOC_READ: libc::c_ulong = 2;
const IOC_TYPE_V: libc::c_ulong = b'V' as libc::c_ulong;

const fn ioc(dir: libc::c_ulong, nr: libc::c_ulong, size: usize) -> libc::c_ulong {
    assert!(size < (1 << 14), "ioctl arg exceeds _IOC_SIZEBITS");
    (dir << 30) | ((size as libc::c_ulong) << 16) | (IOC_TYPE_V << 8) | nr
}

/// One V4L2 request paired with the exact `repr(C)` argument the kernel
/// reads and writes for it. `ioctl` is generic over this trait, so a call
/// can only ever pass the struct that the request code was sized from.
trait IoctlRequest {
    type Arg;
    const CODE: libc::c_ulong;
}

macro_rules! vidioc {
    ($name:ident, $dir:expr, $nr:expr, $arg:ty) => {
        struct $name;
        impl IoctlRequest for $name {
            type Arg = $arg;
            const CODE: libc::c_ulong = ioc($dir, $nr, std::mem::size_of::<$arg>());
        }
    };
}

vidioc!(QueryCap, IOC_READ, 0, V4l2Capability);
vidioc!(SetFormat, IOC_READ | IOC_WRITE, 5, V4l2Format);
vidioc!(RequestBuffers, IOC_READ | IOC_WRITE, 8, V4l2RequestBuffers);
vidioc!(QueryBuffer, IOC_READ | IOC_WRITE, 9, V4l2Buffer);
vidioc!(QueueBuffer, IOC_READ | IOC_WRITE, 15, V4l2Buffer);
vidioc!(DequeueBuffer, IOC_READ | IOC_WRITE, 17, V4l2Buffer);
vidioc!(StreamOn, IOC_WRITE, 18, u32);
vidioc!(StreamOff, IOC_WRITE, 19, u32);

// =============================================================================
// V4L2 Structures (C-compatible; layouts follow <linux/videodev2.h>)
// =============================================================================

// Every struct is plain integer data, so an all-zero value is valid and the
// `Default` impls below replace `mem::zeroed()`. Fields the kernel defines as
// pointer-aligned unions are modelled with explicit padding plus a
// zero-length `usize` array, which gives the C alignment on both 32- and
// 64-bit targets without a Rust `union` (whose reads would be unsafe).

const PTR_PAD: usize = std::mem::size_of::<usize>() - 4;

#[repr(C)]
#[derive(Default)]
struct V4l2Capability {
    driver: [u8; 16],
    card: [u8; 32],
    bus_info: [u8; 32],
    version: u32,
    capabilities: u32,
    device_caps: u32,
    reserved: [u32; 3],
}

#[repr(C)]
#[derive(Default)]
struct V4l2PixFormat {
    width: u32,
    height: u32,
    pixelformat: u32,
    field: u32,
    bytesperline: u32,
    sizeimage: u32,
    colorspace: u32,
    priv_: u32,
    flags: u32,
    ycbcr_enc: u32,
    quantization: u32,
    xfer_func: u32,
}

const V4L2_FORMAT_UNION_BYTES: usize = 200;

// `struct v4l2_format { u32 type; union { v4l2_pix_format pix; ...; u8 raw[200] } fmt; }`
// The union holds pointers in some members, so it is pointer-aligned.
#[repr(C)]
struct V4l2Format {
    type_: u32,
    _pad: [u8; PTR_PAD],
    fmt: V4l2PixFormat,
    _rest: [u8; V4L2_FORMAT_UNION_BYTES - std::mem::size_of::<V4l2PixFormat>()],
    _align: [usize; 0],
}

impl Default for V4l2Format {
    fn default() -> Self {
        Self {
            type_: 0,
            _pad: [0; PTR_PAD],
            fmt: V4l2PixFormat::default(),
            _rest: [0; V4L2_FORMAT_UNION_BYTES - std::mem::size_of::<V4l2PixFormat>()],
            _align: [],
        }
    }
}

#[repr(C)]
#[derive(Default)]
struct V4l2RequestBuffers {
    count: u32,
    type_: u32,
    memory: u32,
    capabilities: u32,
    flags: u8,
    reserved: [u8; 3],
}

// `struct timeval { long tv_sec; long tv_usec; }`
#[repr(C)]
#[derive(Default)]
struct V4l2Timeval {
    tv_sec: libc::c_long,
    tv_usec: libc::c_long,
}

// `union { u32 offset; unsigned long userptr; struct v4l2_plane *planes; s32 fd; } m`
// Every member starts at byte 0, so `offset` is correct on any endianness.
#[repr(C)]
struct V4l2BufferMem {
    offset: u32,
    _pad: [u8; PTR_PAD],
    _align: [usize; 0],
}

impl Default for V4l2BufferMem {
    fn default() -> Self {
        Self {
            offset: 0,
            _pad: [0; PTR_PAD],
            _align: [],
        }
    }
}

#[repr(C)]
#[derive(Default)]
struct V4l2Buffer {
    index: u32,
    type_: u32,
    bytesused: u32,
    flags: u32,
    field: u32,
    timestamp: V4l2Timeval,
    timecode: [u8; 16],
    sequence: u32,
    memory: u32,
    m: V4l2BufferMem,
    length: u32,
    reserved2: u32,
    request_fd: i32,
}

// Sizes pinned to the kernel ABI for the pointer width this is built for.
const _: () = {
    use std::mem::size_of;
    let ptr = size_of::<usize>();
    assert!(size_of::<V4l2Capability>() == 104);
    assert!(size_of::<V4l2PixFormat>() == 48);
    assert!(size_of::<V4l2RequestBuffers>() == 20);
    assert!(size_of::<V4l2Format>() == if ptr == 8 { 208 } else { 204 });
    assert!(size_of::<V4l2Buffer>() == if ptr == 8 { 88 } else { 68 });
};

// =============================================================================
// Memory-Mapped Buffer
// =============================================================================

/// One kernel-owned capture buffer mapped into this process for the life of
/// the value; `Drop` unmaps it.
struct MmapBuffer {
    ptr: *mut u8,
    length: usize,
}

impl MmapBuffer {
    fn map(fd: i32, length: usize, offset: u32) -> Result<Self, CaptureError> {
        // SAFETY: `fd` is the open V4L2 device and `length`/`offset` are
        // what VIDIOC_QUERYBUF reported for an MMAP buffer on that device,
        // so the kernel backs exactly this range. The mapping is owned by
        // the returned value and released in `Drop`.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                libc::off_t::from(offset),
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(CaptureError::CaptureError(format!(
                "mmap failed: {}",
                std::io::Error::last_os_error()
            )));
        }
        Ok(Self {
            ptr: ptr.cast::<u8>(),
            length,
        })
    }

    /// The first `len` bytes of the mapping. Only valid to call while the
    /// buffer is dequeued (between DQBUF and QBUF), when the driver is not
    /// writing to it.
    fn bytes(&self, len: usize) -> &[u8] {
        assert!(
            len <= self.length,
            "V4L2 bytesused {len} exceeds mapped length {}",
            self.length
        );
        // SAFETY: `ptr` is a live MAP_SHARED mapping of `length` bytes owned
        // by `self` (unmapped only in `Drop`), `len <= length` is asserted,
        // and the caller holds the buffer dequeued so nothing writes it
        // while the slice is alive; the slice borrows `self`.
        unsafe { std::slice::from_raw_parts(self.ptr, len) }
    }
}

impl Drop for MmapBuffer {
    fn drop(&mut self) {
        // SAFETY: `ptr`/`length` are exactly what `mmap` returned in `map`,
        // and this is the only place the mapping is released.
        unsafe {
            libc::munmap(self.ptr.cast::<libc::c_void>(), self.length);
        }
    }
}

// =============================================================================
// V4L2 Backend
// =============================================================================

/// V4L2 camera capture backend for Linux.
///
/// Uses memory-mapped I/O for zero-copy frame capture from USB cameras
/// and CSI cameras (Raspberry Pi).
///
/// # Example
/// ```ignore
/// use axonml_vision::camera::{V4L2Backend, CaptureBackend, CaptureConfig, PixelFormat};
///
/// let mut cam = V4L2Backend::new("/dev/video0");
/// cam.open(&CaptureConfig {
///     width: 640,
///     height: 480,
///     format: PixelFormat::Yuyv,
///     fps: 30,
/// }).unwrap();
///
/// let frame = cam.grab_frame().unwrap();
/// cam.close();
/// ```
pub struct V4L2Backend {
    device_path: String,
    file: Option<fs::File>,
    buffers: Vec<MmapBuffer>,
    width: u32,
    height: u32,
    format: PixelFormat,
    streaming: bool,
}

impl V4L2Backend {
    /// Create a V4L2 backend for the specified device.
    ///
    /// Common paths: `/dev/video0`, `/dev/video1`.
    pub fn new(device_path: &str) -> Self {
        Self {
            device_path: device_path.to_string(),
            file: None,
            buffers: Vec::new(),
            width: 0,
            height: 0,
            format: PixelFormat::Yuyv,
            streaming: false,
        }
    }

    /// Open the default camera device (`/dev/video0`).
    pub fn default_device() -> Self {
        Self::new("/dev/video0")
    }

    fn fd(&self) -> Result<i32, CaptureError> {
        self.file
            .as_ref()
            .map(|f| f.as_raw_fd())
            .ok_or(CaptureError::NotOpen)
    }

    fn ioctl<R: IoctlRequest>(&self, arg: &mut R::Arg) -> Result<(), CaptureError> {
        let fd = self.fd()?;
        // SAFETY: `R::CODE` was built from `size_of::<R::Arg>()`, so the
        // kernel copies exactly the bytes of the `repr(C)` struct that
        // `&mut arg` points to (live and exclusively borrowed for the call);
        // the struct layouts are pinned to <linux/videodev2.h> above.
        let ret =
            unsafe { libc::ioctl(fd, R::CODE, std::ptr::from_mut(arg).cast::<libc::c_void>()) };
        if ret < 0 {
            Err(CaptureError::CaptureError(format!(
                "ioctl 0x{:X} failed: {}",
                R::CODE,
                std::io::Error::last_os_error()
            )))
        } else {
            Ok(())
        }
    }

    fn set_format(&mut self, config: &CaptureConfig) -> Result<(), CaptureError> {
        let pixfmt = match config.format {
            PixelFormat::Yuyv => V4L2_PIX_FMT_YUYV,
            PixelFormat::Mjpeg => V4L2_PIX_FMT_MJPEG,
            _ => V4L2_PIX_FMT_YUYV,
        };

        let mut fmt = V4l2Format {
            type_: V4L2_BUF_TYPE_VIDEO_CAPTURE,
            fmt: V4l2PixFormat {
                width: config.width,
                height: config.height,
                pixelformat: pixfmt,
                ..Default::default()
            },
            ..Default::default()
        };

        self.ioctl::<SetFormat>(&mut fmt)?;

        self.width = fmt.fmt.width;
        self.height = fmt.fmt.height;
        self.format = config.format;

        Ok(())
    }

    fn request_buffers(&mut self) -> Result<(), CaptureError> {
        let mut req = V4l2RequestBuffers {
            count: NUM_BUFFERS,
            type_: V4L2_BUF_TYPE_VIDEO_CAPTURE,
            memory: V4L2_MEMORY_MMAP,
            ..Default::default()
        };
        self.ioctl::<RequestBuffers>(&mut req)?;

        let fd = self.fd()?;

        for i in 0..req.count {
            let mut buf = V4l2Buffer {
                index: i,
                type_: V4L2_BUF_TYPE_VIDEO_CAPTURE,
                memory: V4L2_MEMORY_MMAP,
                ..Default::default()
            };
            self.ioctl::<QueryBuffer>(&mut buf)?;

            self.buffers
                .push(MmapBuffer::map(fd, buf.length as usize, buf.m.offset)?);

            // Queue the buffer
            self.ioctl::<QueueBuffer>(&mut buf)?;
        }

        Ok(())
    }

    fn start_streaming(&mut self) -> Result<(), CaptureError> {
        let mut type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        self.ioctl::<StreamOn>(&mut type_)?;
        self.streaming = true;
        Ok(())
    }

    fn stop_streaming(&mut self) {
        if self.streaming {
            let mut type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            let _ = self.ioctl::<StreamOff>(&mut type_);
            self.streaming = false;
        }
    }
}

impl CaptureBackend for V4L2Backend {
    fn open(&mut self, config: &CaptureConfig) -> Result<(), CaptureError> {
        use std::fs::OpenOptions;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.device_path)
            .map_err(|e| CaptureError::DeviceNotFound(format!("{}: {}", self.device_path, e)))?;

        self.file = Some(file);

        // Query capabilities
        let mut cap = V4l2Capability::default();
        self.ioctl::<QueryCap>(&mut cap)?;

        self.set_format(config)?;
        self.request_buffers()?;
        self.start_streaming()?;

        Ok(())
    }

    fn grab_frame(&mut self) -> Result<FrameBuffer, CaptureError> {
        if !self.streaming {
            return Err(CaptureError::NotOpen);
        }

        // Dequeue a buffer
        let mut buf = V4l2Buffer {
            type_: V4L2_BUF_TYPE_VIDEO_CAPTURE,
            memory: V4L2_MEMORY_MMAP,
            ..Default::default()
        };
        self.ioctl::<DequeueBuffer>(&mut buf)?;

        let mmap = self.buffers.get(buf.index as usize).ok_or_else(|| {
            CaptureError::CaptureError(format!("driver returned buffer index {}", buf.index))
        })?;

        // Copy frame data while the buffer is ours (dequeued)
        let frame_data = mmap.bytes(buf.bytesused as usize).to_vec();

        let timestamp_us =
            (buf.timestamp.tv_sec as u64) * 1_000_000 + (buf.timestamp.tv_usec as u64);

        // Re-queue the buffer
        self.ioctl::<QueueBuffer>(&mut buf)?;

        Ok(FrameBuffer {
            data: frame_data,
            width: self.width,
            height: self.height,
            format: self.format,
            timestamp_us,
        })
    }

    fn is_open(&self) -> bool {
        self.streaming
    }

    fn close(&mut self) {
        self.stop_streaming();
        self.buffers.clear();
        self.file = None;
    }

    fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

impl Drop for V4L2Backend {
    fn drop(&mut self) {
        self.close();
    }
}

// Capture itself needs camera hardware and is exercised on target devices;
// the ABI pins below run everywhere.

#[cfg(test)]
mod tests {
    use super::*;

    // Values from <linux/videodev2.h> on a 64-bit target.
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn request_codes_match_kernel_header() {
        assert_eq!(QueryCap::CODE, 0x8068_5600);
        assert_eq!(SetFormat::CODE, 0xC0D0_5605);
        assert_eq!(RequestBuffers::CODE, 0xC014_5608);
        assert_eq!(QueryBuffer::CODE, 0xC058_5609);
        assert_eq!(QueueBuffer::CODE, 0xC058_560F);
        assert_eq!(DequeueBuffer::CODE, 0xC058_5611);
        assert_eq!(StreamOn::CODE, 0x4004_5612);
        assert_eq!(StreamOff::CODE, 0x4004_5613);
    }

    #[test]
    fn buffer_field_offsets_match_kernel_layout() {
        let b = V4l2Buffer::default();
        let base = std::ptr::from_ref(&b) as usize;
        let off = |p: *const u8| p as usize - base;
        let ptr = std::mem::size_of::<usize>();
        assert_eq!(
            off(std::ptr::from_ref(&b.timestamp).cast()),
            if ptr == 8 { 24 } else { 20 }
        );
        assert_eq!(
            off(std::ptr::from_ref(&b.m).cast()),
            if ptr == 8 { 64 } else { 52 }
        );
        assert_eq!(
            off(std::ptr::from_ref(&b.length).cast()),
            if ptr == 8 { 72 } else { 56 }
        );
        let f = V4l2Format::default();
        let fbase = std::ptr::from_ref(&f) as usize;
        assert_eq!(std::ptr::from_ref(&f.fmt) as usize - fbase, ptr.max(4));
    }

    #[test]
    fn mapped_bytes_are_bounded_by_the_mapping() {
        let mut backing = vec![7u8; 16];
        let m = std::mem::ManuallyDrop::new(MmapBuffer {
            ptr: backing.as_mut_ptr(),
            length: backing.len(),
        });
        assert_eq!(m.bytes(4), &[7, 7, 7, 7]);
        assert!(std::panic::catch_unwind(|| m.bytes(17).len()).is_err());
    }
}
