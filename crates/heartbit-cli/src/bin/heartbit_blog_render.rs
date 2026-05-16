//! `heartbit_blog_render` — regenerate the blog static site from
//! Markdown posts. Operator tool for when templates change or
//! posts are edited manually.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use heartbit_ghost::blog::render::{RenderConfig, render_site};

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Regenerate the blog static site from Markdown posts."
)]
struct Args {
    /// Directory holding `*.md` posts.
    #[arg(long, default_value = "blog-site/posts")]
    posts_dir: PathBuf,

    /// Output directory for rendered HTML (this is what gets deployed).
    #[arg(long, default_value = "blog-site/public")]
    out_dir: PathBuf,

    /// Path to style.css (copied verbatim into out_dir).
    #[arg(long, default_value = "blog-site/style.css")]
    style_css: PathBuf,

    /// Public site URL (canonical, sitemap, RSS).
    #[arg(long)]
    site_url: String,

    /// Site title (`<title>` and index header).
    #[arg(long)]
    site_title: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cfg = RenderConfig {
        site_url: &args.site_url,
        site_title: &args.site_title,
        style_css: &args.style_css,
    };
    let metas = render_site(&args.posts_dir, &args.out_dir, &cfg).context("render_site failed")?;
    eprintln!(
        "\u{2713} rendered {} post(s) into {}",
        metas.len(),
        args.out_dir.display()
    );
    for m in &metas {
        eprintln!("  - {} ({})", m.slug, m.date.format("%Y-%m-%d"));
    }
    Ok(())
}
