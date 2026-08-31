import axios from 'axios';
import sharp from 'sharp';

export async function prepareImage(url: string): Promise<{ image: Uint8Array } | { unavailable: string }> {
  const response = await axios.get<ArrayBuffer>(url, {
    responseType: 'arraybuffer', timeout: 30000, maxContentLength: 50 * 1024 * 1024,
    validateStatus: status => (status >= 200 && status < 300) || status === 404 || status === 410,
  });
  if (response.status === 404 || response.status === 410) return { unavailable: `Source image unavailable (HTTP ${response.status}); no visual claims can be made.` };
  // First frame only: animated GIFs can exceed the provider's download limit.
  const image = await sharp(Buffer.from(response.data), { animated: false })
    .rotate().resize({ width: 1536, height: 1536, fit: 'inside', withoutEnlargement: true })
    .jpeg({ quality: 80 }).toBuffer();
  return { image };
}
