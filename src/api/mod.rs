use std::borrow::Cow;

pub type Str<'a> = Cow<'a, str>;
pub type Slice<'a, T> = Cow<'a, [T]>;

pub mod audio;
pub mod common;
pub mod completion;
pub mod edit;
pub mod error;
pub mod image;
pub mod model;
pub mod moderations;

#[inline]
#[allow(unused)]
pub(super) fn trim_ascii(ascii: &[u8]) -> &[u8] {
    return trim_ascii_end(trim_ascii_start(ascii));
}

#[allow(unused)]
pub(super) fn trim_ascii_start(mut ascii: &[u8]) -> &[u8] {
    loop {
        match ascii.first() {
            Some(&x) if x.is_ascii_whitespace() => ascii = &ascii[1..],
            _ => break,
        }
    }
    return ascii;
}

#[allow(unused)]
pub(super) fn trim_ascii_end(mut ascii: &[u8]) -> &[u8] {
    loop {
        match ascii.last() {
            Some(&x) if x.is_ascii_whitespace() => ascii = &ascii[..ascii.len() - 1],
            _ => break,
        }
    }
    return ascii;
}
