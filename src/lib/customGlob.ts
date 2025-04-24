import { promises as fs, existsSync } from 'node:fs';
import { relative } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { bold, green } from 'kleur/colors';
import pLimit from 'p-limit';
// @ts-ignore
import picomatch from 'picomatch';
import { glob as tinyglobby } from 'tinyglobby';
import type { AstroLoader, AstroGlobOptions, AstroLoaderContext } from './custom-types';

interface GenerateIdOptions {
  entry: string;
  base: URL;
  data: Record<string, unknown>;
}

// Simplified ContentEntryType for our custom implementation
interface ContentEntryType {
  getEntryInfo: (options: { contents: string; fileUrl: URL }) => Promise<{ body: string; data: Record<string, unknown> }>;
  getRenderFunction?: (config: any) => Promise<ContentEntryRenderFunction>;
  contentModuleTypes?: unknown;
}

interface ContentEntryRenderFunction {
  (options: { id: string; data: Record<string, unknown>; body: string; filePath: string; digest: string }): Promise<any>;
}

function generateIdDefault({ entry, base, data }: GenerateIdOptions): string {
  if (data.slug) {
    return data.slug as string;
  }
  
//   console.log(`[Custom Glob] Generating ID for entry: ${entry}`);
  
  const entryURL = new URL(encodeURI(entry), base);
  // Simplified version just using the filename without extension as the slug
  const parts = entry.split('/');
  const filename = parts[parts.length - 1];
  const slug = filename.replace(/\.\w+$/, '');
  
//   console.log(`[Custom Glob] Generated slug: ${slug}`);
  
  return slug;
}

function checkPrefix(pattern: string | Array<string>, prefix: string) {
  if (Array.isArray(pattern)) {
    return pattern.some((p) => p.startsWith(prefix));
  }
  return pattern.startsWith(prefix);
}

interface CustomGlobOptions extends AstroGlobOptions {
  generateId?: (options: GenerateIdOptions) => string;
}

export function customGlob(globOptions: CustomGlobOptions): AstroLoader {
  console.log('[Custom Glob] Initializing custom glob loader');
  
  if (checkPrefix(globOptions.pattern, '../')) {
    throw new Error(
      'Glob patterns cannot start with `../`. Set the `base` option to a parent directory instead.'
    );
  }
  if (checkPrefix(globOptions.pattern, '/')) {
    throw new Error(
      'Glob patterns cannot start with `/`. Set the `base` option to a parent directory or use a relative path instead.'
    );
  }

  const generateId = globOptions?.generateId ?? generateIdDefault;
  const fileToIdMap = new Map<string, string>();

  return {
    name: 'custom-glob-loader',
    load: async (context: AstroLoaderContext) => {
      const { config, logger, watcher, parseData, store, generateDigest, entryTypes } = context;
      
      console.log('[Custom Glob] Loading content with custom glob loader');
      
      const renderFunctionByContentType = new Map<ContentEntryType, ContentEntryRenderFunction>();
      const untouchedEntries = new Set(store.keys());
      
      async function syncData(
        entry: string,
        base: URL,
        entryType?: ContentEntryType,
        oldId?: string,
      ) {
        // console.log(`[Custom Glob] Syncing data for entry: ${entry}`);
        
        if (!entryType) {
          logger.warn(`No entry type found for ${entry}`);
          return;
        }
        
        const fileUrl = new URL(encodeURI(entry), base);
        // console.log(`[Custom Glob] Reading file: ${fileUrl.href}`);
        
        const contents = await fs.readFile(fileUrl, 'utf-8').catch((err) => {
          logger.error(`Error reading ${entry}: ${err.message}`);
          return;
        });

        if (!contents && contents !== '') {
          logger.warn(`No contents found for ${entry}`);
          return;
        }

        // console.log(`[Custom Glob] Parsing content for: ${entry}`);
        const { body, data } = await entryType.getEntryInfo({
          contents,
          fileUrl,
        });

        const id = generateId({ entry, base, data });
        // console.log(`[Custom Glob] Generated ID: ${id}`);

        if (oldId && oldId !== id) {
          store.delete(oldId);
        }

        untouchedEntries.delete(id);
        const existingEntry = store.get(id);
        const digest = generateDigest(contents);
        const filePath = fileURLToPath(fileUrl);

        if (existingEntry && existingEntry.digest === digest && existingEntry.filePath) {
          console.log(`[Custom Glob] Entry unchanged: ${id}`);
          
          if (existingEntry.deferredRender) {
            store.addModuleImport(existingEntry.filePath);
          }

          if (existingEntry.assetImports?.length) {
            store.addAssetImports(existingEntry.assetImports, existingEntry.filePath);
          }

          fileToIdMap.set(filePath, id);
          return;
        }

        const relativePath = relative(fileURLToPath(config.root), filePath);
        // console.log(`[Custom Glob] Parsing data for: ${id}`);
        
        const parsedData = await parseData({
          id,
          data,
          filePath,
        });
        
        if (entryType.getRenderFunction) {
          let render = renderFunctionByContentType.get(entryType);
          if (!render) {
            render = await entryType.getRenderFunction(config);
            renderFunctionByContentType.set(entryType, render);
          }
          
          let rendered = undefined;

          try {
            // console.log(`[Custom Glob] Rendering content for: ${id}`);
            rendered = await render?.({
              id,
              data,
              body,
              filePath,
              digest,
            });
          } catch (error: any) {
            logger.error(`Error rendering ${entry}: ${error.message}`);
          }

          store.set({
            id,
            data: parsedData,
            body,
            filePath: relativePath,
            digest,
            rendered,
            assetImports: rendered?.metadata?.imagePaths,
          });
        } else if ('contentModuleTypes' in entryType) {
          store.set({
            id,
            data: parsedData,
            body,
            filePath: relativePath,
            digest,
            deferredRender: true,
          });
        } else {
          store.set({ id, data: parsedData, body, filePath: relativePath, digest });
        }

        fileToIdMap.set(filePath, id);
        // console.log(`[Custom Glob] Successfully processed: ${id}`);
      }

      const baseDir = globOptions.base ? new URL(globOptions.base, config.root) : config.root;

      if (!baseDir.pathname.endsWith('/')) {
        baseDir.pathname = `${baseDir.pathname}/`;
      }

      const filePath = fileURLToPath(baseDir);
      const relativePath = relative(fileURLToPath(config.root), filePath);
      console.log(`[Custom Glob] Base directory: ${relativePath}`);

      const exists = existsSync(baseDir);

      if (!exists) {
        logger.warn(`The base directory "${fileURLToPath(baseDir)}" does not exist.`);
      }

      console.log(`[Custom Glob] Searching for files matching pattern: ${JSON.stringify(globOptions.pattern)}`);
      const files = await tinyglobby(globOptions.pattern, {
        cwd: fileURLToPath(baseDir),
        expandDirectories: false,
      });

      console.log(`[Custom Glob] Found ${files.length} files`);
      
      if (exists && files.length === 0) {
        logger.warn(
          `No files found matching "${globOptions.pattern}" in directory "${relativePath}"`
        );
        return;
      }

      function configForFile(file: string) {
        const ext = file.split('.').at(-1);
        if (!ext) {
          logger.warn(`No extension found for ${file}`);
          return;
        }
        return entryTypes.get(`.${ext}`);
      }

      const limit = pLimit(10);

      await Promise.all(
        files.map((entry) => {
          return limit(async () => {
            const entryType = configForFile(entry);
            await syncData(entry, baseDir, entryType);
          });
        })
      );

      untouchedEntries.forEach((id) => {
        console.log(`[Custom Glob] Removing untouched entry: ${id}`);
        store.delete(id);
      });

      if (!watcher) {
        return;
      }

      watcher.add(filePath);

      const matchesGlob = (entry: string) =>
        !entry.startsWith('../') && picomatch.isMatch(entry, globOptions.pattern);

      const basePath = fileURLToPath(baseDir);

      async function onChange(changedPath: string) {
        const entry = relative(basePath, changedPath).replace(/\\/g, '/');
        if (!matchesGlob(entry)) {
          return;
        }
        console.log(`[Custom Glob] File changed: ${entry}`);
        const entryType = configForFile(changedPath);
        const baseUrl = pathToFileURL(basePath);
        const oldId = fileToIdMap.get(changedPath);
        await syncData(entry, baseUrl, entryType, oldId);
        logger.info(`Reloaded data from ${green(entry)}`);
      }

      watcher.on('change', onChange);
      watcher.on('add', onChange);

      watcher.on('unlink', async (deletedPath: string) => {
        const entry = relative(basePath, deletedPath).replace(/\\/g, '/');
        if (!matchesGlob(entry)) {
          return;
        }
        console.log(`[Custom Glob] File deleted: ${entry}`);
        const id = fileToIdMap.get(deletedPath);
        if (id) {
          store.delete(id);
          fileToIdMap.delete(deletedPath);
        }
      });
    },
  };
} 