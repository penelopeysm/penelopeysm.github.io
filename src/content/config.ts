import { z, defineCollection } from 'astro:content';

const postCollection = defineCollection({
    type: 'content',
    schema: z.object({
        title: z.string(),
        publishDate: z.string().date(),
        sortOrder: z.number().optional(),
    }),
});

export const collections = {
    'posts': postCollection,
}
