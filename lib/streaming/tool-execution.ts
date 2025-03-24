import {
  CoreMessage,
  DataStreamWriter,
  generateId,
  generateText,
  JSONValue
} from 'ai'
import { z } from 'zod'
import { searchSchema } from '../schema/search'
import { search } from '../tools/search'
import { ExtendedCoreMessage } from '../types'
import { getModel } from '../utils/registry'
import { parseToolCallXml } from './parse-tool-call'

interface ToolExecutionResult {
  toolCallDataAnnotations: ExtendedCoreMessage[] | null
  toolCallMessages: CoreMessage[]
}

async function fetchRAGChunks(query: string): Promise<string[]> {
  const response = await fetch('https://api.ragie.ai/retrievals', {
    method: 'POST',
    headers: {
      accept: 'application/json',
      'content-type': 'application/json',
      authorization: `Bearer ${process.env.RAGIE_API_KEY}`

    },
    body: JSON.stringify({ query })
  });

  if (!response.ok) {
    throw new Error(`RAG API error: ${response.status} ${response.statusText}`);
  }

  const json = await response.json();
  return json.scored_chunks.map((chunk: any) => chunk.text); // Extract only the text field
}

export async function executeToolCall(
  coreMessages: CoreMessage[],
  dataStream: DataStreamWriter,
  model: string,
  searchMode: boolean
): Promise<ToolExecutionResult> {
  if (!searchMode) {
    return { toolCallDataAnnotations: null, toolCallMessages: [] }
  }

  const searchSchemaString = Object.entries(searchSchema.shape)
    .map(([key, value]) => {
      const description = value.description
      const isOptional = value instanceof z.ZodOptional
      return `- ${key}${isOptional ? ' (optional)' : ''}: ${description}`
    })
    .join('\n')
  const defaultMaxResults = model?.includes('ollama') ? 5 : 20

  const toolSelectionResponse = await generateText({
    model: getModel(model),
    system: `You are an intelligent assistant that analyzes conversations to select the most appropriate tools and their parameters.
            You excel at understanding context to determine when and how to use available tools, including crafting effective search queries.
            Current date: ${new Date().toISOString().split('T')[0]}

            Do not include any other text in your response.
            Respond in XML format with the following structure:
            <tool_call>
              <tool>tool_name</tool>
              <parameters>
                <query>search query text</query>
                <max_results>number - ${defaultMaxResults} by default</max_results>
                <search_depth>basic or advanced</search_depth>
                <include_domains>domain1,domain2</include_domains>
                <exclude_domains>domain1,domain2</exclude_domains>
              </parameters>
            </tool_call>

            Available tools: search, vedic_rag (use full for any questions related to vedas and Gita)

            Search parameters:
            ${searchSchemaString}

            Vedic RAG parameters:
            ${searchSchemaString}

            If you don't need a tool, respond with <tool_call><tool></tool></tool_call>`,
    messages: coreMessages
  })

  console.log('Tool selection response: ', toolSelectionResponse)

  const toolCall = parseToolCallXml(toolSelectionResponse.text, searchSchema)

  if (!toolCall || toolCall.tool === '') {
    return { toolCallDataAnnotations: null, toolCallMessages: [] }
  }

  const toolCallDataAnnotations: ExtendedCoreMessage[] = []
  const toolCallMessages: CoreMessage[] = []

  // 🔹 VEDIC RAG
  if (toolCall.tool === 'vedic_rag') {
    console.log('vedic tool in invoked, ', toolCall)
    const query = toolCall.parameters?.query ?? ''
    const ragTexts = await fetchRAGChunks(query)
    const formattedResponse = ragTexts.map((text, i) => `Chunk ${i + 1}:\n${text}`).join('\n\n')

    console.log('response', formattedResponse)

    const toolCallAnnotation = {
      type: 'tool_call',
      data: {
        state: 'result',
        toolCallId: `call_${generateId()}`,
        toolName: 'vedic_rag',
        args: JSON.stringify(toolCall.parameters),
        result: JSON.stringify(ragTexts)
      }
    }

    dataStream.writeMessageAnnotation(toolCallAnnotation)

    toolCallDataAnnotations.push({
      role: 'data',
      content: {
        type: 'tool_call',
        data: toolCallAnnotation.data
      } as JSONValue
    })

    toolCallMessages.push({
      role: 'assistant',
      content: `Vedic RAG result:\n${formattedResponse}`
    })
  }

  // 🔹 SEARCH TOOL
  if (toolCall.tool === 'search') {
    const toolCallAnnotation = {
      type: 'tool_call',
      data: {
        state: 'call',
        toolCallId: `call_${generateId()}`,
        toolName: toolCall.tool,
        args: JSON.stringify(toolCall.parameters)
      }
    }

    dataStream.writeData(toolCallAnnotation)

    const searchResults = await search(
      toolCall.parameters?.query ?? '',
      toolCall.parameters?.max_results,
      'basic',
      toolCall.parameters?.include_domains ?? [],
      toolCall.parameters?.exclude_domains ?? []
    )

    const updatedToolCallAnnotation = {
      ...toolCallAnnotation,
      data: {
        ...toolCallAnnotation.data,
        result: JSON.stringify(searchResults),
        state: 'result'
      }
    }

    dataStream.writeMessageAnnotation(updatedToolCallAnnotation)

    toolCallDataAnnotations.push({
      role: 'data',
      content: {
        type: 'tool_call',
        data: updatedToolCallAnnotation.data
      } as JSONValue
    })

    toolCallMessages.push({
      role: 'assistant',
      content: `Search tool result: ${JSON.stringify(searchResults)}`
    })
  }

  // 🔚 Final message prompting the model to proceed
  if (toolCallMessages.length > 0) {
    toolCallMessages.push({
      role: 'user',
      content: 'Now answer the user question using the retrieved knowledge.'
    })
  }

  return {
    toolCallDataAnnotations: toolCallDataAnnotations.length > 0 ? toolCallDataAnnotations : null,
    toolCallMessages
  }
}
