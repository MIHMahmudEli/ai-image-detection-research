/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    domains: [],
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/:path*",
      },
    ];
  },
};

module.exports = nextConfig;
