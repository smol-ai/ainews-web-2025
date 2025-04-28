import rss from "@astrojs/rss";
import { SITE } from "@consts";
import { getCollection } from "astro:content";


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
import { sanitizeString } from "../utils/textUtils";

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
    items: items.map((item) => {
      // Sanitize potential null characters using the utility function
      const title = sanitizeString(item.data.title);
      const description = sanitizeString(item.data.description);
      // Sanitize the date string before converting to Date
      const pubDateStr = sanitizeString(item.data.date);
      const pubDate = new Date(pubDateStr); // Convert sanitized string back to Date

      return {
        title: title,
        description: description,
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
    }),
  });
}
