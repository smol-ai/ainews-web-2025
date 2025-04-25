import React from 'react';

interface OgImageProps {
  title: string;
  description: string;
  formattedDate?: string;
  companyTags?: string[];
  modelTags?: string[];
  issueNumber?: string;
}

export function renderIssueOgImage({
  title,
  description,
  formattedDate,
  companyTags = [],
  modelTags = []
}: OgImageProps) {
  // Get first 10 words of description
  const shortDescription = description.split(' ').slice(0, 10).join(' ') + 
    (description.split(' ').length > 10 ? '...' : '');

  return (
    <div tw="flex flex-col w-full h-full p-12 bg-white font-sans relative">
      {/* Gradient top border */}
      <div tw="absolute top-0 left-0 right-0 h-3 bg-gradient-to-r from-gray-900 to-gray-500" />
      
      {/* Header section with issue info */}
      <div tw="flex justify-between items-center w-full">
        {/* Logo area */}
        <div tw="flex flex-col">
          <span tw="text-2xl font-bold tracking-tight text-gray-800">
            AI NEWS
          </span>
          <span tw="text-sm text-gray-500">
            NEWS.SMOL.AI
          </span>
        </div>
        
        {/* Issue metadata */}
        <div tw="flex flex-col items-end">
          <span tw="text-lg text-gray-600 mt-1">
            {formattedDate}
          </span>
        </div>
      </div>
      
      {/* Divider */}
      <div tw="w-full h-px bg-gray-200 my-0" />
      
      {/* Main content area with title and description */}
      <div tw="flex flex-col flex-1">
        {/* Title */}
        <h1 tw={`font-extrabold tracking-tight text-gray-800 mb-8 max-w-[90%] leading-tight ${
          `text-${Math.max(2, 6 - Math.floor(title.length / 20))}xl`
        }`}>
          {title}
        </h1>
        
        {/* Description (first 10 words) */}
        <p tw="text-2xl text-gray-600 mb-10 leading-normal truncate overflow-hidden text-ellipsis">
          {shortDescription}
        </p>
        
        {/* Tags container */}
        {(companyTags.length > 0 || modelTags.length > 0) && (
          <div tw="flex flex-wrap gap-2">
            {/* Company tags */}
            {companyTags.slice(0, 4).map((tag, i) => (
              <div key={`company-${i}`} tw="bg-green-100 text-green-600 px-3 py-1 rounded-md text-sm">
                {tag}
              </div>
            ))}
            
            {/* Model tags */}
            {modelTags.slice(0, 4).map((tag, i) => (
              <div key={`model-${i}`} tw="bg-indigo-50 text-indigo-600 px-3 py-1 rounded-md text-sm">
                {tag}
              </div>
            ))}
          </div>
        )}
      </div>
      
      {/* Footer */}
      <div tw="flex justify-between items-center mt-4">
        <span tw="text-sm text-gray-500">
          news.smol.ai
        </span>
        <span tw="text-sm text-gray-500">
          @smol_ai
        </span>
      </div>
    </div>
  );
}

export function renderHomepageOgImage({
  title,
  description
}: OgImageProps) {
  // Get first 10 words of description
  const shortDescription = description.split(' ').slice(0, 10).join(' ') + 
    (description.split(' ').length > 10 ? '...' : '');

  return (
    <div tw="flex flex-col w-full h-full p-12 bg-black font-sans">
      {/* Main content area with flexible layout */}
      <div tw="flex flex-col justify-center items-start h-full w-full bg-gray-50 rounded-xl p-16 relative overflow-hidden">
        {/* Logo area */}
        <div tw="flex items-center">
          <span tw="text-xl font-bold text-gray-800">
            AI NEWS
          </span>
          <div tw="w-8 h-8 bg-gray-800 rounded-full flex items-center justify-center mr-3 text-white font-bold">
            🗞️ smol ai
          </div>
        </div>
        
        {/* Headline */}
        <h1 tw="text-7xl font-extrabold tracking-tight text-gray-800 mb-24 max-w-[90%] leading-tight">
          {title}
        </h1>
        
        {/* Description */}
        <p tw="text-2xl text-gray-600 mb-12 leading-normal">
          {shortDescription}
        </p>
        {/* Footer */}
        <div tw="flex w-full justify-between items-center pt-6 mt-4">
          <span tw="text-sm text-gray-500">
            news.smol.ai
          </span>
          <span tw="text-sm text-gray-500">
            @smol_ai
          </span>
        </div>
        {/* Gradient box visual element */}
        <div tw="absolute w-72 h-72 rounded-2xl bg-gradient-to-br from-indigo-100 to-transparent -top-12 -right-12 rotate-12 opacity-70" />
      </div>
    </div>
  );
}
