import rss from "@astrojs/rss";
import { SITE } from "@consts";
import { getCollection } from "astro:content";
import {
  sanitizeString,
  markdownToHtml,
} from "../utils/textUtils";

// catch unsanitary strings

// console.log("--- RSS Items ---");
// items.forEach((item, index) => {
//   // console.log(`Item ${index}:`);
//   // console.log(`  Collection: ${item.collection}`);
//   // console.log(`  ID: ${item.id}`);
//   // console.log(`  Title: ${JSON.stringify(item.data.title)}`);
//   // console.log(`  Description: ${JSON.stringify(item.data.description)}`);
//   // console.log(`  Date: ${JSON.stringify(item.data.date)}`);
//   if (item.data.title?.includes('\u0000') || item.data.description?.includes('\u0000') || item.data.date?.toString().includes('\u0000')) {
//       console.error(`!!! Found potential null character in item ${index} (${item.collection}/${item.id}) !!!`);
//   }
// });
// console.log("--- End RSS Items ---");

export async function GET(context) {
  const issues = (await getCollection("issues")).filter((post) => !post.data.draft);

  const projects = (await getCollection("projects")).filter(
    (project) => !project.data.draft,
  );

  const items = [...issues, ...projects].sort(
    (a, b) => new Date(b.data.date).valueOf() - new Date(a.data.date).valueOf(),
  );


  return rss({
    title: SITE.TITLE,
    description: SITE.DESCRIPTION,
    site: context.site,
    // Map items to RSS format
    items: items.map((item, index) => {
      // Sanitize potential null characters using the utility function
      const title = sanitizeString(item.data.title);
      const descriptionRaw = sanitizeString(item.data.description);
      // Convert description markdown to HTML
      const descriptionHtml = markdownToHtml(descriptionRaw);
      // Sanitize the date string before converting to Date
      const pubDateStr = sanitizeString(item.data.date);
      const pubDate = new Date(pubDateStr); // Convert sanitized string back to Date

      // Prepare the base item object
      const rssItem = {
        title: title,
        description: descriptionHtml, // Use the HTML description
        pubDate: pubDate,
        link: `/${item.collection}/${item.id}/`,
        // Combine all tag-like fields into categories
        categories: [
          ...(item.data.companies?.filter(tag => typeof tag === 'string') ?? []),
          ...(item.data.models?.filter(tag => typeof tag === 'string') ?? []),
          ...(item.data.people?.filter(tag => typeof tag === 'string') ?? []),
          ...(item.data.topics?.filter(tag => typeof tag === 'string') ?? []),
        ],
      };

      // // Add truncated body content for the first 10 items
      // console.log('printing for ' + item.id, item.body)
      if (index < 10 && item.body) {
        const sanitizedBody = sanitizeString(item.body).split('--- # PART 1: High level Discord summaries')[0];
        // Truncate, ensuring we don't leave dangling tags (simple approach)
        const truncatedBody = sanitizedBody.length > 100000
          ? sanitizedBody.substring(0, 100000) + '...'
          : sanitizedBody;
        rssItem.content = markdownToHtml(truncatedBody); // Add the content field
      }

      return rssItem;
    }),
    // (Optional) Add custom data
    customData: `<language>en-us</language>`,
  });
}
