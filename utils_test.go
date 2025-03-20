// utils_test.go
//
// test cases for utils

package gt

import (
	"log"
	"testing"
)

// TestChunkText tests chunking of texts.
func TestChunkText(t *testing.T) {
	const text = `동해물과 백두산이 마르고 닳도록
하느님이 보우하사 우리나라 만세
무궁화 삼천리 화려 강산
대한 사람 대한으로 길이 보전하세

남산 위에 저 소나무 철갑을 두른 듯
바람 서리 불변함은 우리 기상일세
무궁화 삼천리 화려 강산
대한 사람 대한으로 길이 보전하세

가을 하늘 공활한데 높고 구름 없이
밝은 달은 우리 가슴 일편단심일세
무궁화 삼천리 화려 강산
대한 사람 대한으로 길이 보전하세

이 기상과 이 맘으로 충성을 다하여
괴로우나 즐거우나 나라 사랑하세
무궁화 삼천리 화려 강산
대한 사람 대한으로 길이 보전하세
`

	// with no chunking option
	if chunks, err := ChunkText(text); err != nil {
		t.Errorf("failed to chunk text: %s", err)
	} else if _isVerbose {
		log.Printf("chunked texts: %s", prettify(chunks))
	}

	// with some chunking option
	if chunks, err := ChunkText(text, TextChunkOption{
		ChunkSize:      40,
		OverlappedSize: 10,
		EllipsesText:   "...",
	}); err != nil {
		t.Errorf("failed to chunk text: %s", err)
	} else if _isVerbose {
		log.Printf("chunked texts: %s", prettify(chunks))
	}

	// should fail with wrong chunking option
	if _, err := ChunkText(text, TextChunkOption{
		ChunkSize:      40,
		OverlappedSize: 50,
	}); err == nil {
		t.Errorf("should fail with wrong chunking option")
	}
}
